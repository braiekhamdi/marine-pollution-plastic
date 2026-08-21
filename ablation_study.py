"""
ablation_study_v2.py
====================
Revised ablation study with fair YOLO vs SAM
comparison on identical test images.

Replaces ablation_study.py in the repository.
"""

import torch
import numpy as np
import cv2
import os
import glob
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
YOLO_WEIGHTS = (
    #"marine-pollution-plastic"
    "yolov8n_seg_v2/weights/best.pt")
SAM_WEIGHTS  = (
    "sam_models/sam_vit_h_4b8939.pth")

PLASTIC_DIR   = "mydataset/test/plastic"
NOPLASTIC_DIR = "mydataset/test/no-plastic"

DET_IMG = (
    #"/workspaces/marine-pollution-plastic"
    "segmentation_dataset-1/test/images")
DET_LBL = (
   # "/workspaces/marine-pollution-plastic"
    "segmentation_dataset-1/test/labels")

os.makedirs(
    "results/ablation", exist_ok=True)

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def yolo_label_to_mask(lbl_path, h, w):
    mask = np.zeros(
        (h, w), dtype=np.uint8)
    if not os.path.exists(lbl_path):
        return mask
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coords = list(
                map(float, parts[1:]))
            pts = []
            for i in range(
                    0, len(coords)-1, 2):
                pts.append([
                    int(coords[i]*w),
                    int(coords[i+1]*h)])
            if len(pts) >= 3:
                cv2.fillPoly(
                    mask,
                    [np.array(
                        pts,
                        dtype=np.int32)
                     .reshape(-1, 1, 2)],
                    1)
    return mask


def compute_iou(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter/union) \
        if union > 0 else None


# ═════════════════════════════════════════════
# ABLATION 1 — CLASSIFIER GATE
# (unchanged from original — this part
#  was already methodologically correct)
# ═════════════════════════════════════════════
print("\n" + "="*55)
print("ABLATION 1 — Classifier Gate Effect")
print("="*55)

yolo = YOLO(YOLO_WEIGHTS)

# Run YOLO on no-plastic images
no_plastic_files = (
    glob.glob(os.path.join(
        NOPLASTIC_DIR, "*.jpg")) +
    glob.glob(os.path.join(
        NOPLASTIC_DIR, "*.png")))

print(f"Running YOLO on "
      f"{len(no_plastic_files)} "
      f"no-plastic images...")

fp_images    = 0
fp_boxes     = 0

for img_path in no_plastic_files:
    res = yolo(
        img_path, conf=0.25,
        verbose=False)[0]
    if (res.boxes is not None and
            len(res.boxes) > 0):
        fp_images += 1
        fp_boxes  += len(res.boxes)

# Run YOLO on plastic images
plastic_files = (
    glob.glob(os.path.join(
        PLASTIC_DIR, "*.jpg")) +
    glob.glob(os.path.join(
        PLASTIC_DIR, "*.png")))

print(f"Running YOLO on "
      f"{len(plastic_files)} "
      f"plastic images...")

tp_images = 0
for img_path in plastic_files:
    res = yolo(
        img_path, conf=0.25,
        verbose=False)[0]
    if (res.boxes is not None and
            len(res.boxes) > 0):
        tp_images += 1

total    = (len(no_plastic_files) +
            len(plastic_files))
saved_pct = (
    len(no_plastic_files) / total * 100)

print(f"\n--- Ablation 1 Results ---")
print(f"Total test images      : {total}")
print(f"Plastic images         : "
      f"{len(plastic_files)}")
print(f"No-plastic images      : "
      f"{len(no_plastic_files)}")
print(f"\n WITHOUT classifier:")
print(f"  YOLO FP images       : "
      f"{fp_images}")
print(f"  Total false boxes    : "
      f"{fp_boxes}")
print(f"  Unnecessary runs     : "
      f"{len(no_plastic_files)} "
      f"({saved_pct:.1f}%)")
print(f"\n WITH classifier:")
print(f"  YOLO runs on         : "
      f"{len(plastic_files)} images")
print(f"  Processing saved     : "
      f"~{saved_pct:.1f}%")


# ═════════════════════════════════════════════
# ABLATION 2 — FAIR YOLO vs SAM COMPARISON
# KEY CHANGE: both evaluated on SAME images
# in a single loop — no dataset mixing
# ═════════════════════════════════════════════
print("\n" + "="*55)
print("ABLATION 2 — YOLO-only vs YOLO+SAM")
print("(FAIR: same images for both)")
print("="*55)

# Load SAM
use_sam = os.path.exists(SAM_WEIGHTS)
if use_sam:
    try:
        from segment_anything import (
            sam_model_registry,
            SamPredictor)
        sam_model = sam_model_registry[
            "vit_h"](
            checkpoint=SAM_WEIGHTS)
        sam_model.to(device)
        predictor = SamPredictor(sam_model)
        print("SAM loaded successfully.")
    except Exception as e:
        print(f"SAM load error: {e}")
        use_sam = False
else:
    print("SAM weights not found. "
          "Running YOLO-only ablation only.")
    predictor = None

img_files = sorted(
    glob.glob(os.path.join(
        DET_IMG, "*.jpg")) +
    glob.glob(os.path.join(
        DET_IMG, "*.png")))

yolo_ious  = []
sam_ious   = []
skipped    = 0
processed  = 0

print(f"\nProcessing {len(img_files)} "
      f"test images...")

for img_path in img_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB)

    base     = os.path.splitext(
        os.path.basename(img_path))[0]
    lbl_path = os.path.join(
        DET_LBL, base + ".txt")
    gt_mask  = yolo_label_to_mask(
        lbl_path, h, w)

    if gt_mask.sum() == 0:
        skipped += 1
        continue

    # YOLO detection
    res = yolo(
        rgb, conf=0.25,
        verbose=False)[0]

    if (res.boxes is None or
            len(res.boxes) == 0):
        skipped += 1
        continue

    # ── YOLO-only mask ────────────────────
    yolo_mask = np.zeros(
        (h, w), dtype=np.uint8)
    if res.masks is not None:
        for md in res.masks.data:
            m  = md.cpu().numpy()
            mr = cv2.resize(
                m.astype(np.float32),
                (w, h))
            yolo_mask[mr > 0.5] = 1

    iou_y = compute_iou(
        yolo_mask, gt_mask)

    # ── SAM mask (same boxes) ─────────────
    iou_s = None
    if use_sam and predictor is not None:
        predictor.set_image(rgb)
        sam_mask = np.zeros(
            (h, w), dtype=np.uint8)
        boxes = (
            res.boxes.xyxy.cpu().numpy())
        for box in boxes:
            bt = torch.tensor(
                [box], dtype=torch.float,
                device=device)
            tb = (predictor.transform
                  .apply_boxes_torch(
                      bt, rgb.shape[:2]))
            with torch.no_grad():
                masks, sc, _ = (
                    predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=tb,
                        multimask_output=False))
            best = (
                masks[torch.argmax(sc)]
                .squeeze()
                .cpu().numpy())
            sam_mask[best > 0.5] = 1

        iou_s = compute_iou(
            sam_mask, gt_mask)

    # Only count images where BOTH succeed
    if iou_y is not None and \
            (iou_s is not None or
             not use_sam):
        yolo_ious.append(iou_y)
        if iou_s is not None:
            sam_ious.append(iou_s)
        processed += 1
        diff = (f" SAM={iou_s:.4f} "
                f"diff={iou_s-iou_y:+.4f}"
                if iou_s is not None
                else "")
        print(
            f"[{processed:3d}] "
            f"{base[:35]:<35} "
            f"YOLO={iou_y:.4f}"
            f"{diff}")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
y_arr = np.array(yolo_ious)
s_arr = (np.array(sam_ious)
         if len(sam_ious) > 0
         else None)

print(f"\n{'='*55}")
print(f"ABLATION 2 — FAIR RESULTS")
print(f"{'='*55}")
print(f"Images evaluated (both methods): "
      f"{len(yolo_ious)}")
print(f"Skipped (no GT / no detection) : "
      f"{skipped}")
print(f"\nYOLO-only:")
print(f"  Mean IoU  : {y_arr.mean():.4f}")
print(f"  Std IoU   : {y_arr.std():.4f}")
print(f"  IoU≥0.50  : "
      f"{(y_arr>=0.50).mean()*100:.1f}%")
print(f"  IoU≥0.75  : "
      f"{(y_arr>=0.75).mean()*100:.1f}%")

if s_arr is not None:
    print(f"\nYOLO + SAM:")
    print(f"  Mean IoU  : "
          f"{s_arr.mean():.4f}")
    print(f"  Std IoU   : "
          f"{s_arr.std():.4f}")
    print(f"  IoU≥0.50  : "
          f"{(s_arr>=0.50).mean()*100:.1f}%")
    print(f"  IoU≥0.75  : "
          f"{(s_arr>=0.75).mean()*100:.1f}%")
    print(f"\nPer-image comparison:")
    print(f"  SAM better (IoU higher)  : "
          f"{(s_arr > y_arr).sum()} / "
          f"{len(y_arr)} images")
    print(f"  YOLO better (IoU higher) : "
          f"{(y_arr > s_arr).sum()} / "
          f"{len(y_arr)} images")
    print(f"  SAM clearly better (>0.05): "
          f"{(s_arr-y_arr > 0.05).sum()}")
    print(f"  YOLO clearly better (>0.05): "
          f"{(y_arr-s_arr > 0.05).sum()}")


# ─────────────────────────────────────────────
# SAVE TEXT RESULTS
# ─────────────────────────────────────────────
lines = [
    "ABLATION STUDY v2 — RESULTS",
    "="*55,
    "",
    "ABLATION 1 — CLASSIFIER GATE",
    "-"*40,
    f"Total test images    : {total}",
    f"Plastic              : "
    f"{len(plastic_files)}",
    f"No-plastic           : "
    f"{len(no_plastic_files)}",
    f"YOLO FP images       : {fp_images}",
    f"YOLO false boxes     : {fp_boxes}",
    f"Processing saved     : "
    f"{saved_pct:.1f}%",
    "",
    "ABLATION 2 — FAIR YOLO vs SAM",
    "(Both evaluated on same images)",
    "-"*40,
    f"Images evaluated     : "
    f"{len(yolo_ious)}",
    f"YOLO-only Mean IoU   : "
    f"{y_arr.mean():.4f}",
    f"YOLO-only IoU≥0.50   : "
    f"{(y_arr>=0.50).mean()*100:.1f}%",
    f"YOLO-only IoU≥0.75   : "
    f"{(y_arr>=0.75).mean()*100:.1f}%",
]

if s_arr is not None:
    lines += [
        f"SAM       Mean IoU   : "
        f"{s_arr.mean():.4f}",
        f"SAM       IoU≥0.50   : "
        f"{(s_arr>=0.50).mean()*100:.1f}%",
        f"SAM       IoU≥0.75   : "
        f"{(s_arr>=0.75).mean()*100:.1f}%",
        f"SAM better images    : "
        f"{(s_arr > y_arr).sum()} / "
        f"{len(y_arr)}",
        f"YOLO better images   : "
        f"{(y_arr > s_arr).sum()} / "
        f"{len(y_arr)}",
    ]

with open(
    "results/ablation/"
    "ablation_study_v2.txt", "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(lines))
print("\nSaved: results/ablation/"
      "ablation_study_v2.txt")


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(13, 5))
fig.suptitle(
    'Ablation Study Results (v2 — Fair Comparison)',
    fontsize=13, fontweight='bold')

# Ablation 1 plot
labels1 = [
    'Without\nClassifier',
    'With\nClassifier']
vals1   = [total, len(plastic_files)]
bars    = axes[0].bar(
    labels1, vals1,
    color=['coral', 'steelblue'],
    edgecolor='black',
    linewidth=0.8, width=0.5)
for bar, val in zip(bars, vals1):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 3,
        str(val),
        ha='center', va='bottom',
        fontsize=11, fontweight='bold')
axes[0].set_ylabel(
    'Images Sent to YOLO')
axes[0].set_title(
    f'Ablation 1: Classification Filter\n'
    f'({fp_images} FP events blocked)')
axes[0].set_ylim(0, total * 1.2)
axes[0].grid(axis='y', alpha=0.3)

# Ablation 2 plot
if s_arr is not None:
    methods = [
        'YOLO-only\nMasks',
        'YOLO+SAM\nMasks']
    mean_ious = [
        y_arr.mean(), s_arr.mean()]
    i50 = [
        (y_arr>=0.50).mean(),
        (s_arr>=0.50).mean()]
    i75 = [
        (y_arr>=0.75).mean(),
        (s_arr>=0.75).mean()]
    x     = np.arange(2)
    width = 0.25
    for bars_g, vals, lbl, col in [
        (axes[1].bar(
            x-width, mean_ious, width,
            label='Mean IoU',
            color='steelblue',
            edgecolor='black',
            linewidth=0.7),
         mean_ious, 'Mean IoU', 'steelblue'),
        (axes[1].bar(
            x, i50, width,
            label='IoU≥0.50',
            color='darkorange',
            edgecolor='black',
            linewidth=0.7),
         i50, 'IoU≥0.50', 'darkorange'),
        (axes[1].bar(
            x+width, i75, width,
            label='IoU≥0.75',
            color='seagreen',
            edgecolor='black',
            linewidth=0.7),
         i75, 'IoU≥0.75', 'seagreen'),
    ]:
        for bar in bars_g:
            axes[1].text(
                bar.get_x() +
                bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom',
                fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel('Score')
    axes[1].set_title(
        'Ablation 2: YOLO vs SAM\n'
        f'(Fair — same '
        f'{len(yolo_ious)} images each)')
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(
    'results/ablation/'
    'ablation_comparison_v2.png',
    dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: results/ablation/"
      "ablation_comparison_v2.png")
print("\n✅ Ablation study v2 complete!")