# Canine Retinal Degeneration Classification

This repository contains the code to train, evaluate, and interpret deep-learning models for diagnosing canine retinal degeneration (Normal vs. Disease) from fundus photographs, as described in *Smith et al. (2025)*.

---

##  Overview

We compare three architectures using 5-fold cross-validation on the proprietary canine fundus dataset:

- **EfficientNetV2-M** (`tf_efficientnetv2_m.in21k_ft_in1k`)
- **ResNet-50** (`resnet50.a1_in1k`)
- **Vision Transformer (Small, Patch16)** (`vit_small_patch16_384.augreg_in21k_ft_in1k`)

Available scripts:

- **`scripts/train.py`**: Train a single model run and save checkpoints
- **`scripts/eval.py`**: Evaluate a checkpoint on a held-out test set
- **`scripts/crossval.py`**: Perform 5-fold cross-validation with group splits by DogID
- **`scripts/gradcam_demo.py`**: Generate Grad-CAM heatmap overlays for explainability

All scripts use command-line arguments to configure data paths, hyperparameters, and output locations.

---

##  Installation

**Dependencies** (tested with PyTorch, torchvision, timm, scikit-learn, Pillow, tqdm, matplotlib, pandas):

```bash
pip install torch torchvision timm scikit-learn pillow tqdm matplotlib pandas
```

> _You may pin specific versions in your environment or `conda` environment file as needed._

---

##  Repository Structure

```
./
├── scripts/               # Training, evaluation, CV, and Grad-CAM scripts
│   ├── train.py
│   ├── eval.py
│   ├── crossval.py
│   └── gradcam_demo.py
├── src/                   # Shared dataset, model, and utility modules
├── results/               # Default output directory for checkpoints and plots
└── metadata.xlsx          # Metadata file: ImageName (no ext), DogID, etc.
```

---

##  Usage Examples

**1) Single-run training**
```bash
python scripts/train.py \
  --train-dir /path/to/train_images \
  --metadata metadata.xlsx \
  --output-dir results \
  --batch-size 32 \
  --epochs 20
```

**2) Evaluation on held-out test set**
```bash
python scripts/eval.py \
  --checkpoint results/epoch_20_model.pth \
  --test-dir /path/to/test_images \
  --metadata metadata.xlsx \
  --output-dir results
```

**3) 5-Fold Cross-Validation**
```bash
python scripts/crossval.py \
  --data-dir /path/to/train_images \
  --metadata metadata.xlsx \
  --output-dir results \
  --model tf_efficientnetv2_m.in21k_ft_in1k \
  --batch-size 32 \
  --epochs 20 \
  --folds 5
```
Try alternative architectures:
```bash
python scripts/crossval.py --model resnet50.a1_in1k ...
python scripts/crossval.py --model vit_small_patch16_384.augreg_in1k_ft_in1k ...
```

**4) Grad-CAM explainability**
```bash
python scripts/gradcam_demo.py \
  --checkpoint results/epoch_20_model.pth \
  --image /path/to/sample.jpg \
  --output-dir results/gradcam
```

---

## Citation
If you use this code, please cite:

> Smith, D. *et al.* (2025). Deep Learning–Based Diagnosis of Canine Retinal Degeneration Using Fundus Photographs. _Journal of Veterinary Ophthalmology_.

And link to this repository:
```bibtex
@misc{canine_retina_dl_2025,
  author = {Smith, David and Colleagues},
  title = {Canine Retinal Degeneration Classification},
  year = {2025},
  howpublished = {\url{https://github.com/YourOrg/CanineRetinaDL}}
}
```

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or pull request with details.

---

## 📧 Contact
David Smith — smid@upenn.edu

