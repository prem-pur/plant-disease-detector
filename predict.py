# src/predict.py
"""
Single-image inference CLI.
Usage:
    python src/predict.py --image PATH_TO_IMAGE --model model/plant_disease.pth --model_name resnet50 --num_classes 5
"""

import argparse
import torch
from PIL import Image
import numpy as np
import json
from src.model_loader import load_model
from src.utils import get_default_transform, load_class_names, get_device

def main(args):
    device = get_device()
    # Load class names (must exist)
    class_names = load_class_names(args.class_names) if args.class_names else None
    # Load model
    model, device_loaded = load_model(args.model, model_name=args.model_name, num_classes=args.num_classes, device=device)
    # Preprocess image
    img = Image.open(args.image).convert("RGB")
    transform = get_default_transform(image_size=args.image_size)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    pred = class_names[idx] if class_names else str(idx)
    result = {
        "prediction_index": idx,
        "prediction_label": pred,
        "confidence": float(probs[idx]),
        "all_probs": probs.tolist(),
    }
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model", default="model/plant_disease.pth", help="Path to model .pth")
    parser.add_argument("--model_name", default="resnet50", help="resnet50 | efficientnet_b0")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (required for state_dict loading if not inferable)")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--class_names", default="model/class_names.json", help="JSON file containing list of class names")
    args = parser.parse_args()
    main(args)
