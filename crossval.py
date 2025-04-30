# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:16:13 2025

@author: smid
"""




#!/usr/bin/env python3
import os
import argparse
import random
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim.lr_scheduler as lr_scheduler
import timm

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.model_selection import GroupKFold

# ------------------------ 1. Argument Parsing ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="5-Fold CV for canine fundus classifiers")
    p.add_argument("--data-dir",   type=str, default="data/train",
                   help="Root folder with subfolders 0/ and 1/")
    p.add_argument("--metadata",   type=str, default="data/metadata.xlsx",
                   help="Excel file with columns ImageName (no ext) and DogID")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Where to save aggregated metrics and plots")
    p.add_argument("--model",      type=str,
                   choices=[
                     "tf_efficientnetv2_m.in21k_ft_in1k",
                     "resnet50.a1_in1k",
                     "vit_small_patch16_384.augreg_in21k_ft_in1k"
                   ],
                   default="tf_efficientnetv2_m.in21k_ft_in1k",
                   help="Timm model name to cross-validate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--folds",      type=int, default=5)
    p.add_argument("--seed",       type=int, default=221)
    return p.parse_args()

# ------------------------ 2. Global Setup ------------------------
args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ------------------------ 3. Transforms ------------------------
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.1,0.15)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ------------------------ 4. Dataset ------------------------
class CustomDataset(Dataset):
    def __init__(self, root_dir, metadata_df, transform=None):
        self.transform = transform
        # ensure ImageName has no extension
        meta = metadata_df.copy()
        meta["ImageName"] = meta["ImageName"].str.replace(r'\.\w+$','',regex=True)
        self.meta = meta.set_index("ImageName")
        # use ImageFolder for labels & paths
        base = ImageFolder(root_dir)
        self.samples = []
        for path, label in base.samples:
            fname = os.path.basename(path)
            name = os.path.splitext(fname)[0]
            dog_id = self.meta.loc[name, "DogID"]
            self.samples.append((path, label, dog_id, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, dog_id, fname = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, dog_id, fname

# ------------------------ 5. Model Factory ------------------------
def get_model(name, num_classes=2):
    # timm will automatically add a head for num_classes
    return timm.create_model(name, pretrained=True, num_classes=num_classes)

# ------------------------ 6. Training & Eval Fns ------------------------
def train_one_epoch(model, loader, optimizer, scaler, desc):
    model.train()
    running_loss = correct = total = 0
    loader_tq = tqdm(loader, desc=desc)
    for inputs, labels, _, _ in loader_tq:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type=="cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        correct      += preds.eq(labels).sum().item()
        total        += bs
        loader_tq.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return running_loss/total, correct/total

def evaluate(model, loader):
    model.eval()
    losses = total = 0
    all_true = []; all_pred = []; all_scores = []
    with torch.no_grad():
        for inputs, labels, _, _ in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type=="cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scores = torch.softmax(outputs, dim=1)[:,1]
                preds  = outputs.argmax(dim=1)
            bs = labels.size(0)
            losses += loss.item() * bs
            total  += bs
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())
    return losses/total, all_true, all_pred, all_scores

# ------------------------ 7. Cross-Validation Loop ------------------------
if __name__ == "__main__":
    # load metadata once
    meta_df = pd.read_excel(args.metadata)

    # full training set (for both train & validation by fold)
    full_ds     = CustomDataset(args.data_dir, meta_df, transform=train_transform)
    full_ds_val = CustomDataset(args.data_dir, meta_df, transform=val_transform)
    groups = [sample[2] for sample in full_ds.samples]

    gkf = GroupKFold(n_splits=args.folds)
    fold_stats = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(full_ds.samples, groups=groups), start=1):
        print(f"\n=== Fold {fold}/{args.folds} ===")
        train_sub = Subset(full_ds, train_idx)
        val_sub   = Subset(full_ds_val, val_idx)
        train_loader = DataLoader(train_sub, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_sub,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        # instantiate fresh model for this fold
        model = get_model(args.model, num_classes=2).to(DEVICE)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scaler    = torch.cuda.amp.GradScaler()

        # train
        for epoch in range(1, args.epochs+1):
            desc = f"Fold {fold} Ep {epoch}/{args.epochs}"
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, desc)
            scheduler.step()
            print(f" → {desc} — train_loss: {tr_loss:.4f}, train_acc: {tr_acc:.4f}")

        # validate
        val_loss, y_true, y_pred, y_scores = evaluate(model, val_loader)
        acc   = accuracy_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec   = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1    = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        mcc   = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_score = np.nan

        print(f"Fold {fold} VAL — loss: {val_loss:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, mcc: {mcc:.4f}, auc: {auc_score:.4f}")
        fold_stats.append({
            "fold": fold,
            "val_loss": val_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mcc": mcc,
            "kappa": kappa,
            "auc": auc_score
        })

    # ------------------------ 8. Aggregate & Plot ------------------------
    df = pd.DataFrame(fold_stats).set_index("fold")

    def mean_ci(vals):
        m = np.nanmean(vals)
        se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
        ci = 1.96 * se
        return m, (m-ci, m+ci)

    print("\n=== Cross-Validation Summary ===")
    for metric in ["val_loss","accuracy","precision","recall","f1","mcc","kappa","auc"]:
        m, (l, h) = mean_ci(df[metric])
        print(f"{metric:9}: {m:.4f} (95% CI: {l:.4f}–{h:.4f})")

    plt.figure()
    plt.plot(df.index, df["accuracy"], marker="o")
    plt.xlabel("Fold"); plt.ylabel("Accuracy")
    plt.title(f"{args.model} 5-Fold CV Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "cv_accuracy.png"))
    plt.close()
