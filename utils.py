# src/utils.py
"""
Utility helpers: transforms, class name saving/loading, device selection.
"""

import json
from pathlib import Path
import torch
from torchvision import transforms

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_class_names(class_list, out_path="model/class_names.json"):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(class_list, f, indent=2)
    return str(p)

def load_class_names(path="model/class_names.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Create it with save_class_names() or create a json list of class names.")
    return json.load(open(p, "r"))

def get_default_transform(image_size=224):
    """
    Returns torchvision transform used for inference (resize, center crop, normalization).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
