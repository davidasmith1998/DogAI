# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:36:47 2025

@author: smid
"""


#!/usr/bin/env python3
import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

import timm
from PIL import Image
from tqdm import tqdm

# ------------------------ 1. Argparse ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNetV2-M on canine fundus images")
    p.add_argument("--train-dir",   type=str, default="data/train",
                   help="Folder with subfolders 0/ and 1/")
    p.add_argument("--metadata",    type=str, default="data/metadata.xlsx",
                   help="Excel with ImageName (no ext) and DogID")
    p.add_argument("--output-dir",  type=str, default="results",
                   help="Where to save checkpoints & plots")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--seed",        type=int, default=221)
    return p.parse_args()

# ------------------------ 2. Global Config ------------------------
args     = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ------------------------ 3. Custom Dataset ------------------------
class CustomDataset(Dataset):
    def __init__(self, img_dir, metadata_df, transform=None):
        self.transform = transform
        self.meta = metadata_df.set_index("ImageName")
        base = ImageFolder(img_dir)
        self.samples = []
        for path, label in base.samples:
            fname = os.path.basename(path)
            name = os.path.splitext(fname)[0]
            dog_id = self.meta.loc[name, "DogID"]
            self.samples.append((path, label, fname, dog_id))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, fname, dog_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, dog_id, fname

# ------------------------ 4. Transforms ------------------------
train_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.RandomErasing(p=0.5, scale=(0.1,0.15)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ------------------------ 5. Model ------------------------
class EfficientNetV2MClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m.in21k_ft_in1k",
            pretrained=True, num_classes=0, global_pool=""
        )
        in_feats = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.classifier(x)

# ------------------------ 6. Training ------------------------
def plot_training(history, out_dir):
    epochs = history["epoch"]
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["loss"], label="Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history["acc"],  label="Acc");  plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curve.png"))
    plt.close()

def train():
    # load metadata & dataset
    md = pd.read_excel(args.metadata)
    ds = CustomDataset(args.train_dir, md, transform=train_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # model, optimizer, scheduler, scaler
    model = EfficientNetV2MClassifier().to(DEVICE)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    scaler = GradScaler()

    history = {"epoch":[], "loss":[], "acc":[]}
    for ep in range(1, args.epochs+1):
        model.train()
        running_loss = total = correct = 0
        for x,y,_,_ in tqdm(loader, desc=f"Epoch {ep}/{args.epochs}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda", torch.float16, enabled=(DEVICE.type=="cuda")):
                out = model(x)
                loss = criterion(out,y)
                preds = out.argmax(1)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running_loss += loss.item()*y.size(0)
            correct     += preds.eq(y).sum().item()
            total       += y.size(0)
        sched.step()

        epoch_loss = running_loss/total
        epoch_acc  = correct/total
        print(f" → Ep{ep}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        history["epoch"].append(ep)
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

    # save model + curves
    ckpt_path = os.path.join(args.output_dir, f"epoch_{args.epochs}_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    plot_training(history, args.output_dir)
    print("Saved checkpoint to", ckpt_path)

if __name__ == "__main__":
    train()
