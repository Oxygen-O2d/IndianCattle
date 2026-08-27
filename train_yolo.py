"""
train_yolo.py
=============
Fine-tunes YOLOv8n-cls on the 41-class Indian Cattle Breed dataset.

Prerequisites:
    pip install ultralytics
    python split_yolo_dataset.py   (run once to create data_yolo/)

Output:
    best_yolo.pt   — best weights copied to project root

Usage:
    python train_yolo.py
"""

import shutil
from pathlib import Path

# ─── Check ultralytics is installed ───────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.")
    print("Run:  pip install ultralytics")
    raise SystemExit(1)

# ─── Config ───────────────────────────────────────────────────
DATA_DIR   = 'data_yolo'        # created by split_yolo_dataset.py
MODEL_BASE = 'yolov8n-cls.pt'  # nano classification model (~2.7M params)
EPOCHS     = 40
IMG_SIZE   = 224
BATCH      = 64
WORKERS    = 4                  # parallel data loading — safe with __main__ guard
SAVE_PATH  = 'best_yolo.pt'
PROJECT    = str(Path('runs/classify').resolve())  # absolute path avoids nesting
NAME       = 'yolo_experiment'


def main():
    # ─── Sanity check ─────────────────────────────────────────
    if not Path(DATA_DIR).exists():
        print(f"ERROR: '{DATA_DIR}' not found.")
        print("Run:   python split_yolo_dataset.py   first.")
        raise SystemExit(1)

    # ─── Train ────────────────────────────────────────────────
    print("=" * 55)
    print("  BovineAI — YOLOv8n Classification Training")
    print("=" * 55)
    print(f"  Model   : {MODEL_BASE}")
    print(f"  Data    : {DATA_DIR}/")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Img size: {IMG_SIZE}px")
    print(f"  Batch   : {BATCH}")
    print(f"  Workers : {WORKERS}  (0 = Windows-safe single-process loading)")
    print()

    model = YOLO(MODEL_BASE)  # downloads pretrained weights on first run (~7MB)

    model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        patience=10,           # early stopping if no val improvement for 10 epochs
        lr0=0.001,
        lrf=0.01,              # cosine LR decay factor
        warmup_epochs=3,
        # Augmentation — matched to other training scripts
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        verbose=True,
    )

    # ─── Copy best weights to project root ────────────────────
    # Search dynamically — YOLO may create versioned dirs (experiment2, etc.)
    best_candidates = list(Path(PROJECT).rglob('best.pt'))
    if best_candidates:
        best_src = sorted(best_candidates, key=lambda p: p.stat().st_mtime)[-1]
        shutil.copy2(best_src, SAVE_PATH)
        print(f"\n✅ Best weights copied from:\n   {best_src}\n   → {SAVE_PATH}")
    else:
        print(f"\n⚠️  Could not find best.pt under {PROJECT}")

    # ─── Report ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Best model saved : {SAVE_PATH}")
    print("=" * 55)


# ─── REQUIRED on Windows for multiprocessing safety ───────────
if __name__ == '__main__':
    main()
