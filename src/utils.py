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
    data = json.load(open(p, "r"))
    # Accept multiple formats: plain list, or dict containing the list under common keys
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # common key that contains the list of class names
        for k in ("classes", "class_names", "labels", "label_names"):
            if k in data and isinstance(data[k], list):
                return data[k]
        # some checkpoints store a class_to_idx mapping (name -> index)
        if "class_to_idx" in data and isinstance(data["class_to_idx"], dict):
            items = sorted(data["class_to_idx"].items(), key=lambda x: x[1])
            return [k for k, _ in items]
    raise ValueError(f"Unsupported class-names format in {path}. Expected list or dict with 'classes' or 'class_to_idx'.")

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
