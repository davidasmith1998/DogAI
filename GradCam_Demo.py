# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:41:28 2025

@author: smid
"""




#!/usr/bin/env python3
import os
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import numpy as np

# ------------------------ 1. Args ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM on a single image")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image",      type=str, required=True)
    p.add_argument("--output-dir", type=str, default="results")
    return p.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ 2. Transforms ------------------------
transform = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ------------------------ 3. Model + GradCAM ------------------------
class EfficientNetV2MGradCAM(nn.Module):
    def __init__(self):
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
            nn.Linear(512,2)
        )
        self.activations = None
        self.gradients   = None

    @property
    def target_layer(self):
        return self.backbone.conv_head

    def _forward_hook(self, m, inp, out):  self.activations = out
    def _backward_hook(self,m, gi, go):  self.gradients   = go[0]

    def forward(self,x):
        x = self.backbone(x); x = self.pool(x)
        return self.head(x)

    def generate_cam(self, x, cls=None):
        fwd = self.target_layer.register_forward_hook(self._forward_hook)
        bwd = self.target_layer.register_full_backward_hook(self._backward_hook)
        self.eval(); self.zero_grad()
        out = self(x)
        if cls is None: cls = out.argmax(1).item()
        score = out[0,cls]; score.backward(retain_graph=True)
        w = self.gradients.mean(dim=(2,3),keepdim=True)
        cam = F.relu((w*self.activations).sum(1,keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:],mode="bilinear",align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        fwd.remove(); bwd.remove()
        return cam

# ------------------------ 4. Run ------------------------
# load image
img = Image.open(args.image).convert("RGB")
inp = transform(img).unsqueeze(0).to(DEVICE)

# load model
model = EfficientNetV2MGradCAM().to(DEVICE)
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
ckpt = {k.replace("module.",""):v for k,v in ckpt.items()}
model.load_state_dict(ckpt)

# generate CAM
cam = model.generate_cam(inp)

# plot + save
plt.figure(figsize=(6,6))
plt.imshow(img.resize((480,480)))
plt.imshow(cam, cmap="jet", alpha=0.4)
plt.axis("off")
plt.savefig(os.path.join(args.output_dir,"gradcam_overlay.png"), bbox_inches="tight")
