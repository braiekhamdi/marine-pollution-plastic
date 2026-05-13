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

**Key Results**:
- **Classification**: MobileNetV2 achieves **97.21%** test accuracy
- **Detection**: YOLOv8n-seg reaches **mAP@0.50 = 0.6331** on test set
- **Segmentation**: SAM achieves **mean IoU = 0.7922** (86.5% of images > 0.50 IoU)

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

### Option A: KaggleHub (Classification Dataset)

```bash
python3 downloadDataset.py
```

→ This creates a folder `mydataset/` (~1.53 GB) with `train/`, `val/`, `test/` subfolders, each containing `plastic/` and `no-plastic/` classes.

### Option B: Roboflow (Detection + Segmentation Dataset)
> **Note:** Replace the API key in `downloadDatasetRoboflow.py` with your own key from [Roboflow](https://roboflow.com/).

```bash
python3 downloadDatasetRoboflow.py
```

→ This creates a folder `segmentation_dataset-1/` with `data.yaml` and `train/`, `val/`, `test/` splits (images + YOLO‑format labels).

---

## 📥 Model Weights (SAM)

Download the SAM ViT‑H checkpoint:

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

- Outputs:  
  - `models/CustomCNN_best.pth`  
  - `models/MobileNetV2_best.pth`  
- Plots and metrics saved in `results/` (training curves, comparison table)  

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



## 📊 Expected Results  

| Stage              | Model              | Key Metric                  | Value      |
|--------------------|--------------------|-----------------------------|------------|
| Classification     | MobileNetV2        | Test Accuracy               | 97.21%    |
| Detection          | YOLOv8n-seg        | mAP@0.50 (Test)             | 0.6331    |
| Segmentation       | YOLOv8 + SAM ViT-H | Mean IoU (Test)             | 0.7922    |

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

- **Out of memory**: Reduce batch size in training scripts
- **Dataset not found**: Ensure both download scripts ran successfully
- **SAM slow on CPU**: Expected — use GPU if available
- **Roboflow API key**: The provided key works for public dataset

---

## 🤝 Contributing

Contributions welcome! Feel free to open issues or PRs for improvements, new datasets, or deployment optimizations.

---


**Made with ❤️ for environmental conservation and marine ecosystem protection.**

*Let's clean our waters with AI!*


## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{braiek2025deep,
  title={Deep Learning-Based Framework for Plastic Debris Detection in Dynamic Aquatic Ecosystems},
  author={Braiek, Hamdi},
  journal={},
  year={2026}
}
```

## 📧 Contact

- Author: Hamdi Braiek  
- Email: hamdi.houichet@gmail.com  
- GitHub: [braiekhandi](https://github.com/braiekhandi)
---