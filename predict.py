import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None

def load_models(b0_path, cnx_path, class_mapping_path, device):
    if not os.path.exists(class_mapping_path):
        print(f"Error: Could not find {class_mapping_path}.")
        return None, None, None

    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)

    # 1. Load EfficientNet-B0 (Primary)
    model_b0 = None
    if os.path.exists(b0_path):
        model_b0 = models.efficientnet_b0(weights=None)
        model_b0.classifier[1] = nn.Linear(model_b0.classifier[1].in_features, num_classes)
        model_b0.load_state_dict(torch.load(b0_path, map_location=device, weights_only=True))
        model_b0 = model_b0.to(device).eval()

    # 2. Load ConvNeXt-Tiny (Ensemble partner — replaces ResNet-50)
    model_cnx = None
    if os.path.exists(cnx_path):
        model_cnx = models.convnext_tiny(weights=None)
        model_cnx.classifier[2] = nn.Linear(model_cnx.classifier[2].in_features, num_classes)
        model_cnx.load_state_dict(torch.load(cnx_path, map_location=device, weights_only=True))
        model_cnx = model_cnx.to(device).eval()

    return model_b0, model_cnx, class_mapping


def predict_image(image_path, model_b0, model_cnx, class_mapping, device, enable_cam=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        original_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, False

    image_tensor = transform(original_img).unsqueeze(0).to(device)

    # ENSEMBLE: average softmax probabilities from both models
    final_prob = None
    with torch.no_grad():
        if model_b0 is not None:
            prob_b0 = torch.nn.functional.softmax(model_b0(image_tensor)[0], dim=0)
            final_prob = prob_b0

        if model_cnx is not None:
            prob_cnx = torch.nn.functional.softmax(model_cnx(image_tensor)[0], dim=0)
            if final_prob is None:
                final_prob = prob_cnx
            else:
                # Weighted average: ConvNeXt slightly higher weight (64.76% vs 63.66%)
                final_prob = (prob_b0 * 0.49 + prob_cnx * 0.51)

    if final_prob is None:
        print("No models loaded!")
        return None, None, False

    top_prob, top_catid = torch.topk(final_prob, 3)

    results = []
    for i in range(top_prob.size(0)):
        class_idx = str(top_catid[i].item())
        class_name = class_mapping.get(class_idx, "Unknown")
        prob = top_prob[i].item() * 100
        results.append({"breed": class_name, "probability": prob})

    is_ood = top_prob[0].item() < 0.45

    # Grad-CAM always uses EfficientNet-B0 (its features[-1] layer is ideal)
    heatmap_img = None
    if enable_cam and GradCAM is not None and model_b0 is not None:
        target_layers = [model_b0.features[-1]]
        cam = GradCAM(model=model_b0, target_layers=target_layers)
        targets = [ClassifierOutputTarget(top_catid[0].item())]

        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

        img_resized = original_img.resize((256, 256)).crop((16, 16, 240, 240))
        rgb_img = np.float32(img_resized) / 255.0

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        heatmap_img = Image.fromarray(visualization)

    return results, heatmap_img, is_ood
