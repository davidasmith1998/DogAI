# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:40:23 2025

@author: smid
"""


#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import timm
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
)

# ------------------------ 1. Args ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-dir",   type=str, default="data/test")
    p.add_argument("--metadata",   type=str, default="data/metadata.xlsx")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ------------------------ 2. Dataset & Transforms ------------------------
transform = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
md = pd.read_excel(args.metadata).set_index("ImageName")
class CustomDataset:
    # same as in train.py but only returns img, label
    def __init__(self, img_dir, meta, transform):
        self.transform = transform
        base = ImageFolder(img_dir)
        self.samples = []
        for p,l in base.samples:
            fname = os.path.splitext(os.path.basename(p))[0]
            self.samples.append((p,l))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,l = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), l

ds = CustomDataset(args.test_dir, md, transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

# ------------------------ 3. Model Definition ------------------------
class EfficientNetV2MClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m.in21k_ft_in1k",
            pretrained=False, num_classes=0, global_pool=""
        )
        nf = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nf,512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x); x = self.pool(x)
        return self.head(x)

# ------------------------ 4. Load & Eval ------------------------
model = EfficientNetV2MClassifier().to(DEVICE)
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
ckpt = {k.replace("module.",""):v for k,v in ckpt.items()}
model.load_state_dict(ckpt)
model.eval()

y_true = []; y_pred = []; y_score = []; total_loss=0; total=0
with torch.no_grad():
    for x, y in tqdm(loader):
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out,y)
        probs = torch.softmax(out,1)[:,1]
        pred  = out.argmax(1)
        total_loss += loss.item()*y.size(0)
        total      += y.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_score.extend(probs.cpu().tolist())

# metrics
test_loss = total_loss/total
acc  = accuracy_score(y_true,y_pred)
prec = precision_score(y_true,y_pred,average="weighted",zero_division=0)
rec  = recall_score  (y_true,y_pred,average="weighted",zero_division=0)
f1   = f1_score      (y_true,y_pred,average="weighted",zero_division=0)
mcc  = matthews_corrcoef(y_true,y_pred)
kappa= cohen_kappa_score(y_true,y_pred)
try: auc_score = roc_auc_score(y_true,y_score)
except: auc_score=None

# print
print(f"Loss={test_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}, AUC={auc_score}")
print("\nClassification Report:\n", classification_report(y_true,y_pred,zero_division=0))

# save metrics
with open(os.path.join(args.output_dir,"metrics.txt"),"w") as f:
    f.write(f"Loss={test_loss:.4f}\nAcc={acc:.4f}\nF1={f1:.4f}\nMCC={mcc:.4f}\nAUC={auc_score}\n")

# confusion matrix
cm = confusion_matrix(y_true,y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix"); plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir,"confusion_matrix.png"))
