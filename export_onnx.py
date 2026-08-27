import os
import json
import torch
import torch.nn as nn
from torchvision import models

def export_model_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Exporting on device: {device}")

    # Load mapping
    if not os.path.exists('class_mapping.json'):
        print("Missing class_mapping.json")
        return
    with open('class_mapping.json', 'r') as f:
        num_classes = len(json.load(f))

    # Initialize Model geometry
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load PyTorch Weights
    if not os.path.exists('best_model.pth'):
        print("Missing best_model.pth!")
        return
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Create dummy input required for tracing the graph
    print("Tracing the neural graph...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    onnx_path = "best_model.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11,          # Standard opset
        do_constant_folding=True,  # Optimization
        input_names=['input'],     # Define the input name
        output_names=['output'],   # Define the output name
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Successfully exported model to {onnx_path}!")
    print("This C++ optimized graph can natively run on servers, mobile, and IoT devices via ONNXRuntime.")

if __name__ == '__main__':
    export_model_to_onnx()
