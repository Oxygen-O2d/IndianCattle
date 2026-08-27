import os
import json
import warnings

# Suppress PIL Transparency warnings during image loading
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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

def train_model():
    writer = SummaryWriter('runs/efficientnet_experiment')
    
    # 1. Configuration & Hyperparameters
    data_dir = 'data'
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    seed = 42

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # 2. Setup Device (Compute)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 3. Data Augmentation and Normalization
    # Training augmentations to prevent overfitting
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation uses only center crop and normalization
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Load Dataset
    print(f"Loading dataset from '{data_dir}'...")
    full_dataset = datasets.ImageFolder(data_dir)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} breeds.")

    # Save class mapping to JSON for inference
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
        print("Saved class_mapping.json")

    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply respective transforms
    # Note: random_split wraps the dataset, so we apply unique transforms using our global DatasetWrapper

    train_dataset = DatasetWrapper(train_dataset, transform=train_transform)
    val_dataset = DatasetWrapper(val_dataset, transform=val_transform)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training images: {train_size} | Validation images: {val_size}")

    # 6. Initialize Pre-trained Model (EfficientNet-B0)
    print("Loading pre-trained EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace the final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)

    # 7. Loss Function, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 8. Training Loop
    best_val_acc = 0.0
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=correct/total)
            
        epoch_train_loss = running_loss / train_size
        epoch_train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix(loss=loss.item(), acc=val_correct/val_total)
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_correct / val_total
        
        print(f"Summary -> Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch)
        
        # Save Best Model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"-> Saved new best model with Validation Accuracy: {best_val_acc:.4f}")
            
        # Step the scheduler
        scheduler.step(epoch_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        print(f"Current Learning Rate: {current_lr}")

    writer.close()
    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    train_model()
