import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              accuracy_score)
from dataset_loader import (train_loader,
                             val_loader,
                             test_loader,
                             class_names)
from model import (MarineDebrisClassifier,
                   MarineDebrisClassifierV2)
from tqdm import tqdm

# ── 0. Fix seeds for reproducibility ─────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ═════════════════════════════════════════════════════
# TRAINING FUNCTION — works for both models
# ═════════════════════════════════════════════════════
def train_model(model, model_name, lr=0.0001,
                num_epochs=50, patience=5):

    print(f"\n{'='*55}")
    print(f" Training: {model_name}")
    print(f"{'='*55}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=3, factor=0.5)

    # History containers
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0

    for epoch in range(num_epochs):

        # ── Training phase ────────────────────────────
        model.train()
        train_loss = 0
        train_correct = 0
        train_total   = 0

        with tqdm(train_loader,
                  desc=f"[{model_name}] "
                       f"Epoch {epoch+1}/{num_epochs}",
                  unit="batch",
                  leave=False) as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item()
                _, predicted   = torch.max(outputs, 1)
                train_correct += (
                    predicted == labels).sum().item()
                train_total   += labels.size(0)

                pbar.set_postfix(
                    loss=train_loss/(pbar.n+1))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = (train_correct /
                          train_total) * 100

        # ── Validation phase ──────────────────────────
        model.eval()
        val_loss    = 0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (
                    predicted == labels).sum().item()
                val_total   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = (val_correct / val_total) * 100

        # ── Save history ──────────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {avg_val_acc:.2f}%")

        # ── Early stopping ────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            best_epoch       = epoch + 1
            torch.save(
                model.state_dict(),
                f"models/{model_name}_best.pth")
            print(f"  → Best model saved "
                  f"(epoch {best_epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at "
                      f"epoch {epoch+1}. "
                      f"Best epoch: {best_epoch}")
                break

    return model, history, best_epoch


# ═════════════════════════════════════════════════════
# EVALUATION FUNCTION — runs on any loader
# ═════════════════════════════════════════════════════
def evaluate_model(model, loader, split_name):

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(
                predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True)

    acc = accuracy_score(all_labels, all_preds)*100
    cm  = confusion_matrix(all_labels, all_preds)

    print(f"\n--- {split_name} Results ---")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4))
    print(f"Accuracy: {acc:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

    return report, acc, cm


# ═════════════════════════════════════════════════════
# PLOTTING FUNCTION
# ═════════════════════════════════════════════════════
def plot_training_curves(history, model_name,
                         best_epoch):

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Training Curves — {model_name}',
        fontsize=14, fontweight='bold')

    # ── Loss curve ────────────────────────────────────
    axes[0].plot(epochs, history['train_loss'],
                 'b-',  label='Train Loss',
                 linewidth=2)
    axes[0].plot(epochs, history['val_loss'],
                 'r--', label='Validation Loss',
                 linewidth=2)
    axes[0].axvline(x=best_epoch,
                    color='green',
                    linestyle=':',
                    label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy curve ────────────────────────────────
    axes[1].plot(epochs, history['train_acc'],
                 'b-',  label='Train Accuracy',
                 linewidth=2)
    axes[1].plot(epochs, history['val_acc'],
                 'r--', label='Validation Accuracy',
                 linewidth=2)
    axes[1].axvline(x=best_epoch,
                    color='green',
                    linestyle=':',
                    label=f'Best Epoch ({best_epoch})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(
        'Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f'results/{model_name}_training_curves.png',
        dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Curves saved: "
          f"results/{model_name}_training_curves.png")


# ═════════════════════════════════════════════════════
# COMPARISON TABLE FUNCTION
# ═════════════════════════════════════════════════════
def save_comparison_table(results_dict):

    lines = []
    lines.append("="*70)
    lines.append("MODEL COMPARISON TABLE")
    lines.append("(Precision / Recall / F1-score)")
    lines.append("="*70)

    header = (f"{'Model':<22} {'Split':<12} "
              f"{'no-plastic F1':>14} "
              f"{'plastic F1':>12} "
              f"{'Accuracy':>10}")
    lines.append(header)
    lines.append("-"*70)

    for model_name, splits in results_dict.items():
        for split_name, (report, acc) in \
                splits.items():
            np_f1 = report['no-plastic']['f1-score']
            pl_f1 = report['plastic']['f1-score']
            lines.append(
                f"{model_name:<22} "
                f"{split_name:<12} "
                f"{np_f1:>14.4f} "
                f"{pl_f1:>12.4f} "
                f"{acc:>9.2f}%")
        lines.append("-"*70)

    table_str = "\n".join(lines)
    print("\n" + table_str)

    with open("results/model_comparison.txt",
              "w") as f:
        f.write(table_str)
    print("\nTable saved: "
          "results/model_comparison.txt")


# ═════════════════════════════════════════════════════
# MAIN — Train, evaluate, compare both models
# ═════════════════════════════════════════════════════
results_dict = {}

# ── Model 1: Original custom CNN ─────────────────────
model_v1, history_v1, best_ep_v1 = train_model(
    MarineDebrisClassifier(),
    model_name="CustomCNN",
    lr=0.0001,
    num_epochs=50,
    patience=5)

# Load best checkpoint
model_v1.load_state_dict(
    torch.load("models/CustomCNN_best.pth",
               map_location=device))

plot_training_curves(
    history_v1, "CustomCNN", best_ep_v1)

print("\n── CustomCNN Evaluation ──")
results_dict["CustomCNN"] = {}
r, a, _ = evaluate_model(
    model_v1, train_loader, "Train")
results_dict["CustomCNN"]["Train"] = (r, a)

r, a, _ = evaluate_model(
    model_v1, val_loader, "Validation")
results_dict["CustomCNN"]["Validation"] = (r, a)

r, a, cm = evaluate_model(
    model_v1, test_loader, "Test")
results_dict["CustomCNN"]["Test"] = (r, a)

# ── Model 2: MobileNetV2 ──────────────────────────────
model_v2, history_v2, best_ep_v2 = train_model(
    MarineDebrisClassifierV2(),
    model_name="MobileNetV2",
    lr=0.0001,
    num_epochs=50,
    patience=5)

# Load best checkpoint
model_v2.load_state_dict(
    torch.load("models/MobileNetV2_best.pth",
               map_location=device))

plot_training_curves(
    history_v2, "MobileNetV2", best_ep_v2)

print("\n── MobileNetV2 Evaluation ──")
results_dict["MobileNetV2"] = {}
r, a, _ = evaluate_model(
    model_v2, train_loader, "Train")
results_dict["MobileNetV2"]["Train"] = (r, a)

r, a, _ = evaluate_model(
    model_v2, val_loader, "Validation")
results_dict["MobileNetV2"]["Validation"] = (r, a)

r, a, cm = evaluate_model(
    model_v2, test_loader, "Test")
results_dict["MobileNetV2"]["Test"] = (r, a)

# ── Final comparison table ────────────────────────────
save_comparison_table(results_dict)

print("\n✅ All done! Check the results/ folder.")