# Deep Learning-Based Framework for Plastic Debris Detection in Dynamic Aquatic Ecosystems

**Author:** Hamdi Braiek

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0%2B-green.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

---

## 📋 Abstract

Plastic pollution poses a severe threat to aquatic ecosystems. This project implements a **three-stage deep learning pipeline** for detecting, localizing, and segmenting plastic debris in underwater and surface water images:

1. **Binary Classification** — Filter images containing plastic (Custom CNN vs. MobileNetV2)
2. **Object Detection** — Localize debris using **YOLOv8n-seg**
3. **Instance Segmentation** — Refine masks using **Segment Anything Model (SAM) ViT-H** prompted by YOLO boxes

The framework is designed for real-world deployment on boats, drones, or riverbank cameras and emphasizes efficiency, robustness to low-contrast/noisy aquatic conditions, and reproducibility.

---

## 📋 Overview

This repository contains the full implementation of a three-stage deep learning pipeline for detecting, classifying, and segmenting plastic debris in underwater and river optical images.

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│  Stage 1 — Classification   │  MobileNetV2 (transfer learning)
│  Is there plastic?          │  97.21% test accuracy
└──────────────┬──────────────┘
               │ Yes → continue   No → stop
               ▼
┌─────────────────────────────┐
│  Stage 2 — Detection        │  YOLOv8n-seg + CIoU loss
│  Where is the plastic?      │  mAP@0.50 = 0.6331 (test)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3 — Segmentation     │  SAM ViT-H (zero-shot)
│  Pixel-level masks          │  Mean IoU = 0.7922 (test)
└─────────────────────────────┘
```

---

## ✨ Features

- End-to-end pipeline: Classification → Detection → Segmentation
- Classification gatekeeper to reduce unnecessary computation
- YOLOv8 instance segmentation + SAM refinement
- Comprehensive evaluation and ablation studies
- Interactive Streamlit demo
- Reproducible with fixed seeds
- Detailed results, plots, and comparison tables

---

## 🛠️ Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended (but works on CPU)
- ~10 GB free disk space (datasets + models)

### 1. Clone the Repository

```bash
git clone https://github.com/braiekhamdi/marine-pollution-plastic.git
cd marine-pollution-plastic
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `torch`, `torchvision`
- `ultralytics`
- `segment-anything` (from GitHub)
- `streamlit`, `opencv-python`, `pillow-heif`, etc.

---

## 📁 Project Structure

```
marine-pollution-plastic/
├── mydataset/                  # Classification dataset
├── segmentation_dataset-1/     # YOLO detection/seg dataset
├── sam_models/                 # SAM ViT‑H checkpoint
├── models/                     # saved model weights (.pth, .pt)
├── results/                    # Plots, tables, evaluations
├── results/ablation/           # ablation study outputs
├── train.py                    # train both classification models
├── train_yolo_detection.py     # train YOLOv8n‑seg
├── evaluate_yolo.py            # evaluate YOLO on test set
├── evaluate_sam.py             # evaluate SAM segmentation
├── ablation_study.py           # run ablation experiments
├── app3.py                     # Streamlit app
├── downloadDataset.py          # download classification dataset
├── downloadDatasetRoboflow.py  # download detection dataset
├── dataset_loader.py           # PyTorch DataLoader for classification
├── model.py                    # CNN architectures (CustomCNN, MobileNetV2)
├── requirements.txt            # Python dependencies
└── README.md                   # this file
```

---

## 📥 Dataset Download

### A: KaggleHub (Classification Dataset) 
**SouvikDataset** — Marine Plastic Pollution  

```bash
python3 downloadDataset.py
```

→ This creates a folder `mydataset/` (~1.53 GB) with `train/`, `val/`, `test/` subfolders, each containing `plastic/` and `no-plastic/` classes.
- 2,150 underwater images  
- 2 classes: `clean water` and `plastic`  
- License: Open (research use) 

Verify the `mydataset` folder contains:
```
mydataset/
├── train/no-plastic/   (images)
├── train/plastic/      (images)
├── test/no-plastic/    (images)
└── test/plastic/       (images)
```

### B: Roboflow (Detection + Segmentation Dataset)
**Underwater Plastic Segmentation** (Roboflow) 
> **Note:** Replace the API key in `downloadDatasetRoboflow.py` with your own key from [Roboflow](https://roboflow.com/).

```bash
python3 downloadDatasetRoboflow.py
```

→ This creates a folder `segmentation_dataset-1/` with `data.yaml` and `train/`, `val/`, `test/` splits (images + YOLO‑format labels).
- 1,164 annotated images  
- 6 classes: `plastic_bag`, `plastic_bottle`, `plastic_cup`, `face_mask`, `globe`, `plastic_waste`  
- Format: YOLO segmentation (polygon annotations)  
- License: Academic / Non-commercial  
---

Verify the `data.yaml` file contains:
```yaml
train: ../train/images
val:   ../valid/images
test:  ../test/images
nc: 6
names: ['face_mask','globe','plastic_bag',
        'plastic_bottle','plastic_cup',
        'plastic_waste']
```


### Data Split Used

| Split      | Classification | Detection/Segmentation |
|------------|---------------|------------------------|
| Train      | 1,376 images  | 815 images             |
| Validation | 344 images    | 233 images             |
| Test       | 430 images    | 116 images             |
| Ratio      | 80/20 of train split | 70/15/15 |

> ⚠️ The test set was kept completely separate from all training and model selection steps. All final numbers in the paper are reported on the held-out test set only.

Verify Data Loading

```bash
python dataset_loader.py
```

Expected output:
```
Class labels: ['no-plastic', 'plastic']
Train size:      1376
Validation size: 344
Test size:       430
```

## 📥 Model Weights (SAM)

Download manually the SAM ViT‑H checkpoint (file is too large for GitHub: 2.38GB)

```bash
mkdir -p sam_models
wget -P sam_models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---


## 🚀 Pipeline Stages

### Stage 1: Classification

Train both **CustomCNN** and **MobileNetV2** models:

```bash
python3 train.py
```


**Training time:** ~45 minutes on CPU 
- Outputs:  
  - `models/CustomCNN_best.pth`  
  - `models/MobileNetV2_best.pth`  
- Plots and metrics saved in `results/` (training curves, comparison table):
  - `results/CustomCNN_training_curves.png` 
  - `results/MobileNetV2_training_curves.png` 
  - `results/model_comparison.txt`   

**Key configuration:**

| Parameter       | Custom CNN | MobileNetV2 |
|-----------------|-----------|-------------|
| Optimizer       | Adam      | Adam        |
| Learning rate   | 0.0001    | 0.0001      |
| Weight decay    | 1e-4      | 1e-4        |
| Batch size      | 8         | 8           |
| Max epochs      | 50        | 50          |
| Early stopping  | patience=5| patience=5  |
| Best epoch      | 42        | 3           |


**MobileNetV2 is used in the final pipeline.** Test accuracy: **97.21%**

---

### Stage 2: YOLOv8 Detection & Instance Segmentation

Train YOLOv8n-seg on `segmentation_dataset-1`

```bash
python3 train_yolo_detection.py
```
- Trains for 50 epochs (on CPU ≈ 7.5 hours, on GPU much faster)  
- Best weights saved in `marine-pollution-detection/yolov8n_seg_v2/weights/best.pt`  
- Final model copied to `models/yolov8n_trained_final.pt` 

**Key configuration:**

| Parameter         | Value  |
|-------------------|--------|
| Model             | YOLOv8n-seg |
| Epochs            | 50 (best at 49) |
| Optimizer         | SGD    |
| Learning rate     | 0.01   |
| Momentum          | 0.937  |
| Weight decay      | 0.0005 |
| Warmup epochs     | 3      |
| Batch size        | 8      |
| Image size        | 640    |
| Early stopping    | patience=10 |
| Conf threshold    | 0.25   |
| NMS IoU threshold | 0.45   |

**Evaluate on test set:**

```bash
python3 evaluate_yolo.py
```
- Results: `results/yolo_test_results.txt` and per‑class bar chart `results/yolo_test_per_class.png`  
- Achieved **mAP@0.50 = 0.6331** on test set.

---

### Stage 3: SAM Segmentation

Run SAM zero‑shot segmentation using YOLO bounding boxes as prompts:

```bash
python3 evaluate_sam.py
```

- Outputs:  
  - `results/sam_test_results.txt` (mean IoU, thresholds)  
  - `results/sam_iou_distribution.png`  
  - Visual overlays (first 10 images) in `results/sam_eval/`  
- Mean IoU = **0.7922** (86.5% images > 0.50 IoU)

---

## 📊 Ablation Studies

Quantify the contribution of the classification filter and compare YOLO‑only vs YOLO+SAM masks:

```bash
python3 ablation_study.py
```

Evaluates YOLO-only masks vs. YOLO + SAM masks

- Creates `results/ablation/` with:  
  - `ablation1_classifier.txt` – number of images saved / false positives blocked  
  - `ablation2_yolo_vs_sam.txt` – IoU comparison  
  - `ablation_comparison.png` – summary bar chart  

**Key findings:**  
- Classification gatekeeper reduces YOLO processing by **51.2%** and eliminates false positives from clean‑water images.  
- YOLO‑only masks outperform zero‑shot SAM on this dataset (mean IoU 0.8833 vs 0.7922) because YOLO was fine‑tuned on the target domain.


---

## 🌐 Streamlit Demo App

Test the full pipeline interactively:

```bash
python -m streamlit run app3.py
```

The app allows you to:
- Upload an image (or use a sample)
- Run classification → detection → segmentation
- Visualise bounding boxes and SAM masks side‑by‑side

Test the full pipeline on your own images or sample data!
---



## 📊 Results Summary

### Stage 1 — Classification

| Model          | Split | Accuracy | Plastic F1 |
|----------------|-------|----------|------------|
| Custom CNN     | Train | 96.80%   | 0.967      |
| Custom CNN     | Val   | 98.55%   | 0.985      |
| **Custom CNN** | **Test** | **81.16%** | **0.830** |
| MobileNetV2    | Train | 99.64%   | 0.996      |
| MobileNetV2    | Val   | 99.42%   | 0.994      |
| **MobileNetV2** | **Test** | **97.21%** | **0.972** |

### Stage 2 — Detection (Test Set)

| Class          | AP@0.50 | Precision | Recall |
|----------------|---------|-----------|--------|
| plastic_bag    | 0.9437  | 0.9462    | 0.8506 |
| plastic_bottle | 0.8007  | 0.9130    | 0.6316 |
| face_mask      | 0.7873  | 0.8509    | 0.7170 |
| globe          | 0.6392  | 0.6372    | 0.6667 |
| plastic_waste  | 0.6278  | 0.5864    | 0.6304 |
| plastic_cup    | 0.0000  | 0.0000    | 0.0000 |
| **Overall**    | **0.6331** | **0.6556** | **0.5827** |

> `plastic_cup` has only 1 test image — not evaluable, excluded from mean.

### Stage 3 — Segmentation (Test Set)

| Metric            | Value   |
|-------------------|---------|
| Total test images | 116     |
| Images processed  | 104     |
| Skipped (no det.) | 12      |
| **Mean IoU**      | **0.7922** |
| Std IoU           | 0.2577  |
| Min IoU           | 0.0000  |
| Max IoU           | 0.9871  |
| **IoU ≥ 0.50**   | **86.5%** |
| **IoU ≥ 0.75**   | **74.0%** |

### Ablation Study

| Config                         | YOLO Input | FP Events | Mean IoU | IoU≥0.50 |
|-------------------------------|-----------|-----------|----------|----------|
| No filter + YOLO-only         | 430        | 98        | 0.8833   | 96.4%    |
| No filter + YOLO+SAM          | 430        | 98        | 0.7922   | 86.5%    |
| **Full pipeline + YOLO-only** | **210**    | **0**     | **0.7944** | **88.5%** |
| Full pipeline + YOLO+SAM      | 210        | 0         | 0.7922   | 86.5%    |

### End-to-End Pipeline Analysis

| Stage                  | Value  | Cumulative Rate |
|------------------------|--------|-----------------|
| 1 — Classification     | 0.9905 | 0.9905          |
| 2 — Detection          | 0.5827 | 0.5772          |
| 3 — Coverage           | 0.8966 | 0.5175          |
| 3 — IoU ≥ 0.50         | 0.8650 | **0.4476**      |

> End-to-end success rate: **~44.9%** of plastic images → correct classification → detection → mask with IoU ≥ 0.50.

See `results/` folder for full tables, confusion matrices, and plots.

---

## 🖼️ Visualizations

The pipeline produces:
- Training curves
- Per-class performance charts
- Detection + segmentation overlays
- IoU distributions
- Ablation comparison plots

---


## 🔧 Troubleshooting
**`ReduceLROnPlateau: unexpected keyword argument 'verbose'`**  
Remove the `verbose=True` argument from the scheduler — it was removed in newer PyTorch versions.

**`No module named 'segment_anything'`**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**`pin_memory warning`**  
This is a non-critical warning from PyTorch when no GPU is available. It does not affect results.

**SAM takes very long (~40 seconds/image)**  
This is expected on CPU. SAM ViT-H is a large transformer model. On a GPU it runs in ~0.2 seconds. Use YOLO built-in masks for faster inference.

**`faster-coco-eval` auto-installs during evaluation**  
This is handled automatically by Ultralytics. Re-run the script after it installs.

---

## 🤝 Contributing

Contributions welcome! Feel free to open issues or PRs for improvements, new datasets, or deployment optimizations.

---


*Made with ❤️ for environmental conservation and marine ecosystem protection.*

*Let's clean our waters with AI!*


## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{braiek2026deep,
  title={Deep Learning-Based Framework for Plastic Debris Detection in Dynamic Aquatic Ecosystems},
  author={Braiek, Hamdi},
  journal={},
  year={2026}
}
```

## 📧 Contact

- Author: Hamdi Braiek  
- Email: hamdi.houichet@gmail.com  
- GitHub: [braiekhanmdi](https://github.com/braiekhamdi)
---
