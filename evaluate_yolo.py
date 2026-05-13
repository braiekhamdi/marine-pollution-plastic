from ultralytics import YOLO
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# ── Load your trained model ────────────────────────────
# Try best weights first, fallback to last
model_path = (
    "yolov8n_seg_v2/weights/best.pt")

if not os.path.exists(model_path):
    model_path = (
        "yolov8n_seg_v2/weights/last.pt")
    print(f"best.pt not found, using last.pt")

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# ── Run evaluation on TEST set ─────────────────────────
print("\n" + "="*55)
print("YOLO EVALUATION — TEST SET")
print("="*55)

metrics = model.val(
    data=(
        'segmentation_dataset-1/data.yaml'),
    split='test',      # ← test set only
    conf=0.25,
    iou=0.45,
    verbose=True,
    save_json=True,
    project='results',
    name='yolo_test_eval',
    exist_ok=True)

# ── Extract and print key metrics ─────────────────────
print("\n" + "="*55)
print("SUMMARY — TEST SET RESULTS")
print("="*55)

mp   = metrics.box.mp          # mean precision
mr   = metrics.box.mr          # mean recall
map50     = metrics.box.map50  # mAP@0.50
map5095   = metrics.box.map    # mAP@0.50:0.95

print(f"Overall Precision  : {mp:.4f}")
print(f"Overall Recall     : {mr:.4f}")
print(f"mAP@0.50           : {map50:.4f}")
print(f"mAP@0.50:0.95      : {map5095:.4f}")

# ── Per-class results ──────────────────────────────────
print("\n--- Per-Class Results ---")

names = model.names  # class name dictionary
ap_per_class   = metrics.box.ap          # AP@0.50:0.95
ap50_per_class = metrics.box.ap50        # AP@0.50
p_per_class    = metrics.box.p           # precision
r_per_class    = metrics.box.r           # recall

print(f"\n{'Class':<20} {'Precision':>10} "
      f"{'Recall':>10} {'AP@0.50':>10} "
      f"{'AP@0.50:0.95':>14}")
print("-"*65)

for i, name in names.items():
    try:
        print(f"{name:<20} "
              f"{p_per_class[i]:>10.4f} "
              f"{r_per_class[i]:>10.4f} "
              f"{ap50_per_class[i]:>10.4f} "
              f"{ap_per_class[i]:>14.4f}")
    except IndexError:
        print(f"{name:<20} {'N/A':>10}")

print("-"*65)
print(f"{'Overall (mean)':<20} "
      f"{mp:>10.4f} "
      f"{mr:>10.4f} "
      f"{map50:>10.4f} "
      f"{map5095:>14.4f}")

# ── Save results to text file ──────────────────────────
os.makedirs("results", exist_ok=True)

with open("results/yolo_test_results.txt",
          "w") as f:
    f.write("YOLO TEST SET EVALUATION\n")
    f.write("="*65 + "\n")
    f.write(f"Model: {model_path}\n\n")
    f.write(f"Overall Precision : {mp:.4f}\n")
    f.write(f"Overall Recall    : {mr:.4f}\n")
    f.write(f"mAP@0.50          : {map50:.4f}\n")
    f.write(f"mAP@0.50:0.95     : {map5095:.4f}\n\n")
    f.write(f"{'Class':<20} {'Precision':>10} "
            f"{'Recall':>10} {'AP@0.50':>10} "
            f"{'AP@0.50:0.95':>14}\n")
    f.write("-"*65 + "\n")
    for i, name in names.items():
        try:
            f.write(
                f"{name:<20} "
                f"{p_per_class[i]:>10.4f} "
                f"{r_per_class[i]:>10.4f} "
                f"{ap50_per_class[i]:>10.4f} "
                f"{ap_per_class[i]:>14.4f}\n")
        except IndexError:
            f.write(f"{name:<20} {'N/A':>10}\n")

print("\nResults saved to: "
      "results/yolo_test_results.txt")

# ── Plot per-class AP@0.50 bar chart ───────────────────
# ── Plot per-class AP@0.50 bar chart ──────────────────
# Filter out classes with zero AP (e.g. plastic_cup)
valid_classes  = []
valid_ap50     = []
valid_prec     = []
valid_recall   = []

for i, name in names.items():
    try:
        ap_val = float(ap50_per_class[i])
        p_val  = float(p_per_class[i])
        r_val  = float(r_per_class[i])

        # Only include classes with
        # at least some detections
        if ap_val > 0.0 or p_val > 0.0 \
                or r_val > 0.0:
            valid_classes.append(name)
            valid_ap50.append(ap_val)
            valid_prec.append(p_val)
            valid_recall.append(r_val)
        else:
            print(f"Skipping '{name}' "
                  f"— no detections "
                  f"in test set.")
    except IndexError:
        print(f"Skipping '{name}' "
              f"— index error.")

print(f"\nPlotting {len(valid_classes)} "
      f"classes: {valid_classes}")

try:
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5))
    fig.suptitle(
        'YOLOv8n-seg Detection Performance'
        ' — Test Set '
        f'({len(valid_classes)} classes)',
        fontsize=13,
        fontweight='bold')

    # ── Bar chart — AP@0.50 per class ─────────────────
    colors = plt.cm.Set2(
        np.linspace(
            0, 1, len(valid_classes)))

    bars = axes[0].bar(
        valid_classes,
        valid_ap50,
        color=colors,
        edgecolor='black',
        linewidth=0.7)

    # Mean line — computed only on valid classes
    mean_ap50_valid = np.mean(valid_ap50)
    axes[0].axhline(
        y=mean_ap50_valid,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Mean AP@0.50 '
              f'= {mean_ap50_valid:.3f}')

    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('AP@0.50')
    axes[0].set_title(
        'Per-Class AP@0.50 on Test Set')
    axes[0].set_ylim(0, 1.10)
    axes[0].legend()
    axes[0].tick_params(
        axis='x', rotation=30)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, valid_ap50):
        axes[0].text(
            bar.get_x() +
            bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold')

    # ── Bar chart — Precision vs Recall ───────────────
    x     = np.arange(len(valid_classes))
    width = 0.35

    axes[1].bar(
        x - width / 2,
        valid_prec,
        width,
        label='Precision',
        color='steelblue',
        edgecolor='black',
        linewidth=0.7)

    axes[1].bar(
        x + width / 2,
        valid_recall,
        width,
        label='Recall',
        color='coral',
        edgecolor='black',
        linewidth=0.7)

    # Add value labels on bars
    for val, xpos in zip(
            valid_prec,
            x - width / 2):
        axes[1].text(
            xpos, val + 0.01,
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=8)

    for val, xpos in zip(
            valid_recall,
            x + width / 2):
        axes[1].text(
            xpos, val + 0.01,
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=8)

    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Score')
    axes[1].set_title(
        'Per-Class Precision & Recall'
        ' on Test Set')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        valid_classes,
        rotation=30,
        ha='right')
    axes[1].set_ylim(0, 1.15)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        'results/yolo_test_per_class.png',
        dpi=150,
        bbox_inches='tight')
    plt.close()
    print("\nPlot saved: "
          "results/yolo_test_per_class.png")
    print(f"Classes shown: "
          f"{valid_classes}")
    excluded = [
        n for n in names.values()
        if n not in valid_classes]
    print(f"Classes excluded (zero AP): "
          f"{excluded}")

except Exception as e:
    print(f"Plot error (non-critical): {e}")