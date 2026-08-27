# 🐂 BovineAI: Indian Cattle Breed Classification & Benchmarking Suite

**BovineAI** is an end-to-end deep learning framework and diagnostic suite designed for high-granularity identification of **41 Indigenous Indian Cattle Breeds** (e.g., Gir, Kankrej, Sahiwal, Hallikar, Ongole). 

The project includes a modern **CustomTkinter Desktop Application**, a **Multi-Architecture Benchmark Engine** (ConvNeXt-Tiny, EfficientNet-B0, YOLOv8n-cls, ResNet-50), and an **Automated Research Reporting Pipeline** featuring Grad-CAM explainability heatmaps and executive PDF generation.

---

## 📊 Benchmark Summary (41 Breeds)

| Architecture | Model Variant | Parameters | Top-1 Accuracy | Paradigm | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny** | Modernized CNN | **28.0 M** | **64.76%** | High Accuracy | SOTA Cloud Diagnostic |
| **EfficientNet-B0** | Compound Scaling | **5.3 M** | **63.66%** | Balanced | Fast Desktop Inference |
| **YOLOv8n-cls** | CSP Real-time | **2.7 M** | **56.91%** | Edge Efficient | Mobile & Field Deployment |
| **ResNet-50** | Classic Residual | **25.6 M** | **43.25%** | Reference | Research Baseline |

*Evaluated on a strict 20% validation split across 5,926 high-resolution image samples.*

---

## 🖥️ CustomTkinter Desktop Application

`gui.py` provides a state-of-the-art dark-mode desktop GUI built with **CustomTkinter** and `tkinterdnd2`.

### Features:
- 📥 **Drag-and-Drop Diagnostic Interface**: Instantly load and classify cattle images with live progress animations.
- ⚡ **Ensemble Prediction Engine**: Blends ConvNeXt-Tiny & EfficientNet-B0 model predictions with confidence scoring and Top-3 breed rankings.
- 📂 **Batch Directory Processing**: Analyze entire image directories and export batch summary logs.
- 📑 **PDF Executive Report Export**: Export publication-ready diagnostic reports (`BovineAI_Research_Report.pdf`).
- 📚 **Breed Encyclopedia & NIM Audit**: Interactive morphological guide for indigenous cattle breeds.

To launch the desktop application:
```bash
python gui.py
```

---

## 🛠️ Project Setup & Installation

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (e.g., RTX 3050 6GB Laptop GPU or higher recommended)

### 2. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/Oxygen-O2d/IndianCattle.git
cd IndianCattle
pip install -r requirements.txt
```

> **Note for PyTorch GPU Support:** If required for your CUDA setup:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 📁 Repository Structure

```text
IndianCattle/
├── gui.py                        # CustomTkinter Desktop Application GUI
├── predict.py                    # Multi-model ensemble inference pipeline
├── train.py                      # Training script for EfficientNet-B0
├── train_convnext.py             # Training script for ConvNeXt-Tiny
├── train_resnet.py               # Training script for ResNet-50
├── train_yolo.py                 # Training script for YOLOv8n-cls
├── split_yolo_dataset.py         # Formats raw dataset into YOLO train/val structure
├── generate_paper_figures.py     # Generates publication PDF, ROC curves, confusion matrix, Grad-CAM
├── evaluate.py                   # Evaluation script for validation set
├── class_mapping.json            # Class index to breed name mapping (41 breeds)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Version control ignore configuration
└── README.md                     # Documentation
```

---

## 🚀 Model Training & Evaluation

### Train Individual Architectures:
- **EfficientNet-B0:** `python train.py`
- **ConvNeXt-Tiny:** `python train_convnext.py`
- **ResNet-50:** `python train_resnet.py`
- **YOLOv8n-cls:** `python split_yolo_dataset.py` then `python train_yolo.py`

### CLI Inference:
```bash
python predict.py path/to/cattle_image.jpg
```

---

## 🔬 Research & Paper Figures Generator

Generate the full research publication PDF report along with 300 DPI figure artifacts (Confusion Matrices, Grad-CAM heatmaps, ROC curves, and training metrics):

```bash
python generate_paper_figures.py
```

Outputs:
- `BovineAI_Research_Report.pdf`
- High-resolution figures saved in `research_artifacts/`

---

## 📜 Citation & License

This project is developed as part of research into indigenous livestock conservation and automated agricultural diagnostics. 

*Developed by the BovineAI Research Team.*
