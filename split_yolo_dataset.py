"""
split_yolo_dataset.py
=====================
Creates a YOLO-compatible train/val folder structure from the existing
flat `data/<breed>/<images>` directory.

Output layout:
    data_yolo/
        train/<breed>/<images>
        val/<breed>/<images>

Uses the same 80/20 split with seed=42 as all other training scripts
so the comparison is fair (identical val set).

Usage:
    python split_yolo_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

DATA_SRC  = Path('data')
DATA_YOLO = Path('data_yolo')
SEED      = 42
VAL_RATIO = 0.20

random.seed(SEED)

if DATA_YOLO.exists():
    print(f"'{DATA_YOLO}' already exists — deleting and recreating...")
    shutil.rmtree(DATA_YOLO)

total_train, total_val = 0, 0
breeds = sorted([d for d in DATA_SRC.iterdir() if d.is_dir()])
print(f"Found {len(breeds)} breed classes.\n")

for breed_dir in breeds:
    images = list(breed_dir.glob('*.jpg')) + \
             list(breed_dir.glob('*.jpeg')) + \
             list(breed_dir.glob('*.png'))
    random.shuffle(images)

    val_count   = max(1, int(len(images) * VAL_RATIO))
    val_imgs    = images[:val_count]
    train_imgs  = images[val_count:]

    for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
        dest = DATA_YOLO / split / breed_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dest / img.name)

    total_train += len(train_imgs)
    total_val   += len(val_imgs)
    print(f"  {breed_dir.name:<35} train={len(train_imgs):>3}  val={len(val_imgs):>3}")

print(f"\n✅ Split complete!")
print(f"   Total train: {total_train}")
print(f"   Total val  : {total_val}")
print(f"   Output dir : {DATA_YOLO.resolve()}")
