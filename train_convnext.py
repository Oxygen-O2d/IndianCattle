"""
train_convnext.py
=================
Fine-tunes ConvNeXt-Tiny on the 41-class Indian Cattle Breed dataset.

Key differences from ResNet training:
  - AdamW optimizer (matches ConvNeXt pre-training regime)
  - Lower LR (1e-4) with Cosine Annealing
  - Label smoothing (reduces overconfident predictions)
  - 40 epochs for full convergence
  - Saves to best_convnext.pth

Usage:
    python train_convnext.py
"""

import os
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR       = 'data'
SAVE_PATH      = 'best_convnext.pth'
TB_RUN         = 'runs/convnext_experiment'
BATCH_SIZE     = 32
NUM_EPOCHS     = 40
LR             = 1e-4          # ConvNeXt prefers a lower LR than ResNet
WEIGHT_DECAY   = 0.01          # AdamW standard
LABEL_SMOOTH   = 0.1           # smooths overconfident predictions
SEED           = 42


# ─────────────────────────────────────────────────────────────
# DATASET WRAPPER
# ─────────────────────────────────────────────────────────────
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def train_convnext():
    writer = SummaryWriter(TB_RUN)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Transforms ───────────────────────────────────────────
    # ConvNeXt benefits from stronger augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Dataset ──────────────────────────────────────────────
    print(f"Loading dataset from '{DATA_DIR}'...")
    full_ds = datasets.ImageFolder(DATA_DIR)
    class_names  = full_ds.classes
    num_classes  = len(class_names)
    print(f'Classes : {num_classes}')

    # Save / overwrite class mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)

    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_raw, val_raw = torch.utils.data.random_split(
        full_ds, [train_size, val_size])

    train_ds = DatasetWrapper(train_raw, transform=train_transform)
    val_ds   = DatasetWrapper(val_raw,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train: {train_size}  |  Val: {val_size}')

    # ── Model ────────────────────────────────────────────────
    print('Loading ConvNeXt-Tiny (ImageNet pretrained)...')
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # Replace the head — ConvNeXt classifier is Sequential[LayerNorm, Flatten, Linear]
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable')

    # ── Loss, Optimizer, Scheduler ───────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # AdamW — the optimizer ConvNeXt was originally trained with
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    # Cosine annealing — smooth LR decay over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── Training Loop ────────────────────────────────────────
    best_val_acc = 0.0
    print(f'\nStarting training for {NUM_EPOCHS} epochs...\n')

    for epoch in range(NUM_EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        bar = tqdm(train_loader,
                   desc=f'Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]')
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping — stabilises ConvNeXt fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
            bar.set_postfix(loss=f'{loss.item():.3f}',
                            acc=f'{correct/total:.3f}')

        epoch_train_loss = run_loss / train_size
        epoch_train_acc  = correct / total

        # ---------- VALIDATE ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            vbar = tqdm(val_loader,
                        desc=f'Epoch {epoch+1:02d}/{NUM_EPOCHS} [Val]  ')
            for inputs, labels in vbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * inputs.size(0)
                _, preds     = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += (preds == labels).sum().item()
                vbar.set_postfix(acc=f'{val_correct/val_total:.3f}')

        epoch_val_loss = val_loss / val_size
        epoch_val_acc  = val_correct / val_total
        current_lr     = optimizer.param_groups[0]['lr']

        print(f'  Summary -> Train {epoch_train_acc:.4f} ({epoch_train_loss:.4f}) '
              f'| Val {epoch_val_acc:.4f} ({epoch_val_loss:.4f}) '
              f'| LR {current_lr:.2e}')

        # TensorBoard
        writer.add_scalar('Loss/train',       epoch_train_loss, epoch)
        writer.add_scalar('Loss/val',         epoch_val_loss,   epoch)
        writer.add_scalar('Accuracy/train',   epoch_train_acc,  epoch)
        writer.add_scalar('Accuracy/val',     epoch_val_acc,    epoch)
        writer.add_scalar('Learning_Rate',    current_lr,       epoch)

        # Save best
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  ✓ New best saved → {SAVE_PATH}  '
                  f'(val acc = {best_val_acc:.4f})')

        scheduler.step()

    writer.close()
    print(f'\n{"="*50}')
    print(f'Training complete!')
    print(f'Best Validation Accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)')
    print(f'Model saved to           : {SAVE_PATH}')
    print(f'{"="*50}')


if __name__ == '__main__':
    train_convnext()
