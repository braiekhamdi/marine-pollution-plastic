import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import (DataLoader,
                               random_split)

# ── Paths ─────────────────────────────────────────────
TRAIN_DIR = "mydataset/train"
TEST_DIR  = "mydataset/test"

# ── Transforms ────────────────────────────────────────
# Training transform — with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# Validation and test transform — NO augmentation
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# ── Load full train dataset ────────────────────────────
# We load it twice:
# once with augmentation  → for training
# once without            → for validation split
full_train_aug  = ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform)

full_train_eval = ImageFolder(
    root=TRAIN_DIR,
    transform=eval_transform)

# ── Split: 80% train / 20% validation ─────────────────
total_size = len(full_train_aug)
val_size   = int(0.20 * total_size)
train_size = total_size - val_size

# Fix seed so split is always the same
generator = torch.Generator().manual_seed(42)

train_indices, val_indices = random_split(
    range(total_size),
    [train_size, val_size],
    generator=generator)

# Use indices to create subsets
from torch.utils.data import Subset

train_dataset = Subset(
    full_train_aug,
    train_indices)

val_dataset = Subset(
    full_train_eval,   # no augmentation
    val_indices)

# ── Test dataset ───────────────────────────────────────
test_dataset = ImageFolder(
    root=TEST_DIR,
    transform=eval_transform)  # no augmentation

# ── DataLoaders ────────────────────────────────────────
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False)

# ── Class names ────────────────────────────────────────
class_names = full_train_aug.classes
print(f"Class labels: {class_names}")
print(f"Train size:      {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size:       {len(test_dataset)}")

# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    for images, labels in train_loader:
        print(f"Train batch — "
              f"Images: {images.shape}, "
              f"Labels: {labels}")
        break
    for images, labels in val_loader:
        print(f"Val batch — "
              f"Images: {images.shape}, "
              f"Labels: {labels}")
        break
    for images, labels in test_loader:
        print(f"Test batch — "
              f"Images: {images.shape}, "
              f"Labels: {labels}")
        break


