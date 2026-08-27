import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

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

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load validation dataset (must match train.py splitting logic)
    data_dir = 'data'
    seed = 42
    batch_size = 32

    torch.manual_seed(seed)
    full_dataset = datasets.ImageFolder(data_dir)
    
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
        
    num_classes = len(class_mapping)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Use global DatasetWrapper
    val_dataset = DatasetWrapper(val_dataset, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load Model
    print("Loading best_model.pth...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. Predict on Validation Set
    all_preds = []
    all_labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Generate Classification Report
    class_names = [class_mapping[str(i)] for i in range(num_classes)]
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # 5. Generate and Plot Confusion Matrix
    print("Saving Confusion Matrix plot to 'confusion_matrix.png'...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Validation Confusion Matrix', fontsize=24)
    plt.ylabel('True Breed', fontsize=18)
    plt.xlabel('Predicted Breed', fontsize=18)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Done!")

if __name__ == '__main__':
    evaluate_model()
