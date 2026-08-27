# Indian Cattle Breed Classification

This project uses a dataset of Indian cattle breeds to train an image classification model using PyTorch. 
It fine-tunes an `EfficientNet-B0` model which provides an excellent balance of high accuracy and low memory footprint, which is perfect for RTX 3050 laptops (6GB VRAM).

## Setup

1. **Ensure you have Anaconda or a Virtual Environment.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > *Note:* If you face issues with PyTorch GPU support, install the specific CUDA 12 version:
   > `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` 
   > (PyTorch standard builds currently support 12.1/12.4, which works perfectly with native CUDA 12.9)

## Directory Structure
Ensure your `data` directory contains one subfolder per breed BEFORE training.
```text
IndianCattle/
├── data/
│   ├── Gir/
│   ├── Kankrej/
│   └── ... (41 breeds total)
├── requirements.txt
├── train.py
├── predict.py
└── README.md
```

## Training

To train the model, simply run:
```bash
python train.py
```
This script handles the 80/20 train/validation split automatically.
It saves the best performing weights as `best_model.pth` and a mapping dictionary as `class_mapping.json`.

## Inference

To predict the breed of a new image you downloaded or have, run:
```bash
python predict.py path/to/your/image.jpg
```
It will output the top 3 most likely breeds with their confidence percentage.
