from ultralytics import YOLO
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt

# ── Fix seeds for reproducibility ─────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs("results", exist_ok=True)

# ── Device ─────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset path ───────────────────────────────────────
DATA_YAML = (
    'segmentation_dataset-1/data.yaml')

# ═══════════════════════════════════════════════════════
# STEP 1 — TRAIN
# ═══════════════════════════════════════════════════════
def train():
    print("\n" + "="*55)
    print(" TRAINING YOLOv8n-seg")
    print("="*55)

    model = YOLO('yolov8n-seg.pt')

    results = model.train(
        data=DATA_YAML,

        # ── Core settings ──────────────────────────────
        epochs=50,          # more epochs than before
        imgsz=640,
        batch=8,            # safe for CPU/low memory
        seed=42,            # reproducibility

        # ── Optimizer ──────────────────────────────────
        optimizer='SGD',    # SGD more stable than Adam
        lr0=0.01,           # initial learning rate
        lrf=0.01,           # final lr = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,    # warmup for first 3 epochs
        warmup_momentum=0.8,

        # ── Augmentation ───────────────────────────────
        hsv_h=0.015,        # hue augmentation
        hsv_s=0.7,          # saturation augmentation
        hsv_v=0.4,          # value augmentation
        flipud=0.3,         # vertical flip prob
        fliplr=0.5,         # horizontal flip prob
        mosaic=1.0,         # mosaic augmentation
        scale=0.5,          # image scale augmentation
        translate=0.1,      # translation augmentation

        # ── Early stopping ─────────────────────────────
        patience=10,        # stop if no improvement
                            # for 10 epochs

        # ── Saving and logging ─────────────────────────
        save=True,
        save_period=5,      # save every 5 epochs
        val=True,           # validate every epoch
        plots=True,         # save training plots

        # ── Hardware ───────────────────────────────────
        device=device,
        workers=2,          # safe for Codespaces

        # ── Project ────────────────────────────────────
        project='marine-pollution-detection',
        name='yolov8n_seg_v2',
        exist_ok=True,
        verbose=True,
    )

    # Save final model
    model.save("models/yolov8n_trained_final.pt")
    print("\nTraining complete. Model saved.")
    return results


# ═══════════════════════════════════════════════════════
# STEP 2 — EVALUATE ON ALL THREE SPLITS
# ═══════════════════════════════════════════════════════
def evaluate_all_splits():

    # Load best weights from training
    best_weights = (
        'marine-pollution-detection'
        '/yolov8n_seg_v2/weights/best.pt')

    if not os.path.exists(best_weights):
        best_weights = (
            'marine-pollution-detection'
            '/yolov8n_seg_v2/weights/last.pt')
        print(f"best.pt not found, using last.pt")

    print(f"\nLoading best model: {best_weights}")
    model = YOLO(best_weights)

    # ── Evaluate each split ────────────────────────────
    splits_results = {}
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*55}")
        print(f" EVALUATING ON: {split.upper()}")
        print(f"{'='*55}")

        try:
            metrics = model.val(
                data=DATA_YAML,
                split=split,
                conf=0.25,
                iou=0.45,
                verbose=False,
                project='results',
                name=f'eval_{split}',
                exist_ok=True)

            splits_results[split] = metrics
            print(
                f"{split.upper()} | "
                f"P={metrics.box.mp:.4f} | "
                f"R={metrics.box.mr:.4f} | "
                f"mAP50={metrics.box.map50:.4f} | "
                f"mAP50-95="
                f"{metrics.box.map:.4f}")

        except Exception as e:
            print(f"Error on split {split}: {e}")

    return model, splits_results


# ═══════════════════════════════════════════════════════
# STEP 3 — PRINT AND SAVE FULL TABLE
# ═══════════════════════════════════════════════════════
def save_full_table(model, splits_results):

    names = model.names
    class_names_list = [
        names[i] for i in range(len(names))]

    lines = []
    lines.append("="*75)
    lines.append(
        "YOLOv8n-seg PERFORMANCE TABLE "
        "— TRAIN / VALIDATION / TEST")
    lines.append("="*75)

    for split, metrics in splits_results.items():

        lines.append(
            f"\n--- {split.upper()} SET ---")
        lines.append(
            f"{'Class':<18} "
            f"{'Precision':>10} "
            f"{'Recall':>10} "
            f"{'AP@0.50':>10} "
            f"{'AP@0.50:0.95':>14}")
        lines.append("-"*65)

        p   = metrics.box.p
        r   = metrics.box.r
        ap50     = metrics.box.ap50
        ap5095   = metrics.box.ap

        for i, name in enumerate(
                class_names_list):
            try:
                lines.append(
                    f"{name:<18} "
                    f"{p[i]:>10.4f} "
                    f"{r[i]:>10.4f} "
                    f"{ap50[i]:>10.4f} "
                    f"{ap5095[i]:>14.4f}")
            except IndexError:
                lines.append(
                    f"{name:<18} "
                    f"{'N/A':>10}")

        lines.append("-"*65)
        lines.append(
            f"{'Overall (mean)':<18} "
            f"{metrics.box.mp:>10.4f} "
            f"{metrics.box.mr:>10.4f} "
            f"{metrics.box.map50:>10.4f} "
            f"{metrics.box.map:>14.4f}")

    table_str = "\n".join(lines)
    print("\n" + table_str)

    # Save to file
    with open(
            "results/yolo_full_table.txt",
            "w") as f:
        f.write(table_str)

    print("\nTable saved: "
          "results/yolo_full_table.txt")


# ═══════════════════════════════════════════════════════
# STEP 4 — PLOT COMPARISON CHART
# ═══════════════════════════════════════════════════════
def plot_comparison(model, splits_results):

    names  = model.names
    labels = [names[i]
              for i in range(len(names))]
    x      = np.arange(len(labels))
    width  = 0.25

    colors = {
        'train': 'steelblue',
        'val':   'darkorange',
        'test':  'seagreen'}

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6))
    fig.suptitle(
        'YOLOv8n-seg Performance '
        'Across All Splits',
        fontsize=14,
        fontweight='bold')

    # ── AP@0.50 per class per split ────────────────────
    for idx, (split, metrics) in enumerate(
            splits_results.items()):
        ap50 = metrics.box.ap50
        axes[0].bar(
            x + idx * width,
            ap50[:len(labels)],
            width,
            label=split.capitalize(),
            color=colors[split],
            edgecolor='black',
            linewidth=0.6,
            alpha=0.85)

    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('AP@0.50')
    axes[0].set_title(
        'Per-Class AP@0.50 by Split')
    axes[0].set_xticks(
        x + width)
    axes[0].set_xticklabels(
        labels, rotation=30, ha='right')
    axes[0].set_ylim(0, 1.15)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # ── Overall mAP@0.50 summary ───────────────────────
    split_names = list(splits_results.keys())
    map50_vals  = [
        splits_results[s].box.map50
        for s in split_names]
    map5095_vals = [
        splits_results[s].box.map
        for s in split_names]

    x2    = np.arange(len(split_names))
    w2    = 0.35
    bars1 = axes[1].bar(
        x2 - w2/2,
        map50_vals,
        w2,
        label='mAP@0.50',
        color='steelblue',
        edgecolor='black',
        linewidth=0.7)
    bars2 = axes[1].bar(
        x2 + w2/2,
        map5095_vals,
        w2,
        label='mAP@0.50:0.95',
        color='coral',
        edgecolor='black',
        linewidth=0.7)

    # Add value labels
    for bar in bars1 + bars2:
        axes[1].text(
            bar.get_x() +
            bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=9)

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(
        [s.capitalize()
         for s in split_names])
    axes[1].set_ylabel('mAP Score')
    axes[1].set_title(
        'Overall mAP Across Splits')
    axes[1].set_ylim(0, 1.15)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        'results/yolo_comparison.png',
        dpi=150,
        bbox_inches='tight')
    plt.close()
    print("Plot saved: "
          "results/yolo_comparison.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":

    # Step 1 — Train
    train()

    # Step 2 — Evaluate all splits
    model, splits_results = evaluate_all_splits()

    # Step 3 — Print and save table
    save_full_table(model, splits_results)

    # Step 4 — Plot comparison
    plot_comparison(model, splits_results)

    print("\n✅ All done! "
          "Check results/ folder.")