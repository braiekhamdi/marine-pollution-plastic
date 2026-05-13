import torch
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import time
from segment_anything import (sam_model_registry,
                               SamPredictor)
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────
YOLO_WEIGHTS = (
    "yolov8n_seg_v2/weights/best.pt")

SAM_WEIGHTS  = (
    "sam_models/sam_vit_h_4b8939.pth")

TEST_IMG_DIR = (
    "/segmentation_dataset-1/test/images")

TEST_LBL_DIR = (
    "/segmentation_dataset-1/test/labels")

os.makedirs("results/sam_eval", exist_ok=True)

# ── Device ─────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════
# HELPER — Convert YOLO polygon label to binary mask
# ══════════════════════════════════════════════════════
def yolo_label_to_mask(label_path,
                        img_h, img_w):
    """
    Reads a YOLO segmentation label file
    and returns a binary mask of shape
    (img_h, img_w).
    YOLO seg format per line:
    class x1 y1 x2 y2 ... (normalized)
    """
    mask = np.zeros(
        (img_h, img_w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # Skip class index (parts[0])
        coords = list(map(float, parts[1:]))

        # coords are: x1 y1 x2 y2 ...
        # normalized to [0,1]
        points = []
        for i in range(
                0, len(coords) - 1, 2):
            x = int(coords[i]   * img_w)
            y = int(coords[i+1] * img_h)
            points.append([x, y])

        if len(points) < 3:
            continue

        pts = np.array(
            points,
            dtype=np.int32).reshape(
                (-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ══════════════════════════════════════════════════════
# HELPER — Compute IoU between two binary masks
# ══════════════════════════════════════════════════════
def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union
    between two binary numpy arrays.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    intersection = np.logical_and(
        pred, gt).sum()
    union        = np.logical_or(
        pred, gt).sum()

    if union == 0:
        return None  # skip empty masks

    return float(intersection) / float(union)


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
print("\nLoading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading SAM model (vit_h)...")
sam = sam_model_registry["vit_h"](
    checkpoint=SAM_WEIGHTS)
sam.to(device)
predictor = SamPredictor(sam)
print("Models loaded.\n")


# ══════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════
image_files = sorted(glob.glob(
    os.path.join(TEST_IMG_DIR, "*.jpg")) +
    glob.glob(
    os.path.join(TEST_IMG_DIR, "*.png")))

print(f"Found {len(image_files)} "
      f"test images.\n")

iou_scores       = []
skipped_no_det   = 0
skipped_no_gt    = 0
skipped_empty_gt = 0
processed        = 0
inference_times  = []

for idx, img_path in enumerate(image_files):

    # ── Load image ────────────────────────
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        continue

    img_h, img_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB)

    # ── Load ground truth mask ────────────
    base = os.path.splitext(
        os.path.basename(img_path))[0]
    lbl_path = os.path.join(
        TEST_LBL_DIR, base + ".txt")

    gt_mask = yolo_label_to_mask(
        lbl_path, img_h, img_w)

    if gt_mask.sum() == 0:
        skipped_empty_gt += 1
        continue

    # ── YOLO detection ────────────────────
    t_start = time.time()

    yolo_results = yolo_model(
        image_rgb,
        conf=0.25,
        verbose=False)[0]

    if (yolo_results.boxes is None or
            len(yolo_results.boxes) == 0):
        skipped_no_det += 1
        continue

    boxes = (yolo_results.boxes
             .xyxy.cpu().numpy())

    # ── SAM segmentation ──────────────────
    predictor.set_image(image_rgb)

    boxes_tensor = torch.tensor(
        boxes, device=device)
    transformed_boxes = (
        predictor.transform
        .apply_boxes_torch(
            boxes_tensor,
            image_rgb.shape[:2]))

    with torch.no_grad():
        masks_pred, scores, _ = (
            predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False))

    t_end = time.time()
    inference_times.append(
        t_end - t_start)

    # ── Combine all predicted masks ───────
    combined_pred = np.zeros(
        (img_h, img_w), dtype=np.uint8)

    for mask in masks_pred:
        m = mask.squeeze().cpu().numpy()
        combined_pred[m > 0.5] = 1

    # ── Compute IoU ───────────────────────
    iou = compute_iou(combined_pred, gt_mask)

    if iou is not None:
        iou_scores.append(iou)
        processed += 1

        print(f"[{idx+1:3d}/{len(image_files)}]"
              f" {base[:40]:<40} "
              f"IoU={iou:.4f}")

    # ── Save visual for first 10 images ───
    if processed <= 10:
        vis = image_bgr.copy()

        # Draw GT mask in green
        gt_overlay = np.zeros_like(image_bgr)
        gt_overlay[gt_mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(
            vis, 0.7,
            gt_overlay, 0.3, 0)

        # Draw predicted mask in red
        pred_overlay = np.zeros_like(
            image_bgr)
        pred_overlay[
            combined_pred > 0] = (0, 0, 255)
        vis = cv2.addWeighted(
            vis, 0.8,
            pred_overlay, 0.2, 0)

        save_path = os.path.join(
            "results/sam_eval",
            f"{base}_eval.jpg")
        cv2.imwrite(save_path, vis)


# ══════════════════════════════════════════════════════
# COMPUTE FINAL METRICS
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("SAM SEGMENTATION — TEST SET RESULTS")
print("="*55)

if len(iou_scores) == 0:
    print("No IoU scores computed. "
          "Check your paths and labels.")
else:
    iou_arr = np.array(iou_scores)

    mean_iou  = np.mean(iou_arr)
    std_iou   = np.std(iou_arr)
    iou_50    = np.mean(iou_arr >= 0.50)*100
    iou_75    = np.mean(iou_arr >= 0.75)*100
    avg_time  = np.mean(inference_times)*1000

    print(f"Total test images     : "
          f"{len(image_files)}")
    print(f"Images processed      : "
          f"{processed}")
    print(f"Skipped (no detection): "
          f"{skipped_no_det}")
    print(f"Skipped (empty GT)    : "
          f"{skipped_empty_gt}")
    print(f"\nMean IoU              : "
          f"{mean_iou:.4f}")
    print(f"Std IoU               : "
          f"{std_iou:.4f}")
    print(f"IoU >= 0.50 (%)       : "
          f"{iou_50:.1f}%")
    print(f"IoU >= 0.75 (%)       : "
          f"{iou_75:.1f}%")
    print(f"Min IoU               : "
          f"{np.min(iou_arr):.4f}")
    print(f"Max IoU               : "
          f"{np.max(iou_arr):.4f}")
    print(f"Avg inference time    : "
          f"{avg_time:.1f} ms/image")

    # ── Save results to file ──────────────
    with open(
            "results/sam_test_results.txt",
            "w") as f:
        f.write("SAM SEGMENTATION EVALUATION"
                " — TEST SET\n")
        f.write("="*55 + "\n")
        f.write(
            f"SAM variant        : vit_h\n"
            f"YOLO weights       : "
            f"{YOLO_WEIGHTS}\n"
            f"Total images       : "
            f"{len(image_files)}\n"
            f"Processed images   : "
            f"{processed}\n"
            f"Skipped (no det)   : "
            f"{skipped_no_det}\n"
            f"Skipped (empty GT) : "
            f"{skipped_empty_gt}\n\n"
            f"Mean IoU           : "
            f"{mean_iou:.4f}\n"
            f"Std IoU            : "
            f"{std_iou:.4f}\n"
            f"IoU >= 0.50 (%)    : "
            f"{iou_50:.1f}%\n"
            f"IoU >= 0.75 (%)    : "
            f"{iou_75:.1f}%\n"
            f"Min IoU            : "
            f"{np.min(iou_arr):.4f}\n"
            f"Max IoU            : "
            f"{np.max(iou_arr):.4f}\n"
            f"Avg inference time : "
            f"{avg_time:.1f} ms/image\n")

    print("\nResults saved: "
          "results/sam_test_results.txt")

    # ── Plot IoU distribution ─────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5))
    fig.suptitle(
        'SAM Segmentation — Test Set '
        'IoU Distribution',
        fontsize=13,
        fontweight='bold')

    # Histogram
    axes[0].hist(
        iou_arr,
        bins=20,
        color='steelblue',
        edgecolor='black',
        alpha=0.85)
    axes[0].axvline(
        mean_iou,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean IoU = {mean_iou:.3f}')
    axes[0].axvline(
        0.50,
        color='orange',
        linestyle=':',
        linewidth=1.5,
        label='IoU = 0.50 threshold')
    axes[0].set_xlabel('IoU Score')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title(
        'IoU Score Distribution')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Cumulative distribution
    sorted_iou = np.sort(iou_arr)
    cumulative  = (
        np.arange(1, len(sorted_iou)+1)
        / len(sorted_iou) * 100)

    axes[1].plot(
        sorted_iou,
        cumulative,
        color='steelblue',
        linewidth=2)
    axes[1].axvline(
        0.50,
        color='orange',
        linestyle=':',
        linewidth=1.5,
        label=f'IoU=0.50 → '
              f'{iou_50:.1f}% of images')
    axes[1].axvline(
        0.75,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'IoU=0.75 → '
              f'{iou_75:.1f}% of images')
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel(
        'Cumulative % of Images')
    axes[1].set_title(
        'Cumulative IoU Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        'results/sam_iou_distribution.png',
        dpi=150,
        bbox_inches='tight')
    plt.close()
    print("Plot saved: "
          "results/sam_iou_distribution.png")