# src/model_loader.py
"""
Utilities to load a PyTorch model saved as:
 - full model object (torch.save(model, path))
 - or a state_dict (torch.save(model.state_dict(), path))
This file tries to auto-detect which kind of file you have.
Adjust MODEL_NAME and NUM_CLASSES as appropriate for your trained model.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
MODEL_NAME = "resnet50"  # change to 'efficientnet_b0' if you trained EfficientNet
NUM_CLASSES = None  # if None, will try to infer from saved dict (if possible)

def build_architecture(model_name: str, num_classes: int):
    """Return a fresh model architecture (untrained) matching the training backbone."""
    if model_name == "resnet50":
        base = models.resnet50(pretrained=False)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        return base
    elif model_name == "efficientnet_b0":
        base = models.efficientnet_b0(pretrained=False)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, num_classes)
        return base
    else:
        raise ValueError(f"Unsupported MODEL_NAME {model_name}")

def load_model(pth_path: str, model_name: str = MODEL_NAME, num_classes: int = NUM_CLASSES, device: str = None):
    """
    Loads model from pth_path.
    - If the file contains a state_dict (mapping of parameter tensors), you must provide num_classes.
    - If it contains a full model object, it will be returned directly.
    Returns: model (torch.nn.Module) set to eval(), device used.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pth = Path(pth_path)
    if not pth.exists():
        raise FileNotFoundError(f"{pth_path} not found")

    logging.info(f"Loading {pth_path} on device {device} ...")
    data = torch.load(pth_path, map_location="cpu")

    # case 1: full model object (nn.Module)
    if isinstance(data, nn.Module):
        model = data
        model.to(device)
        model.eval()
        logging.info("Loaded full model object from .pth")
        return model, device

    # case 2: state_dict or checkpoint dict
    if isinstance(data, dict):
        # Guess whether it's a checkpoint with keys like 'model_state_dict'
        if "model_state_dict" in data and isinstance(data["model_state_dict"], dict):
            state_dict = data["model_state_dict"]
            # try to infer num_classes from final layer size if possible
            if num_classes is None:
                # attempt to find fc weight param and infer out_features
                for k, v in state_dict.items():
                    if k.endswith(".fc.weight") or ".classifier.1.weight" in k:
                        out_features = v.shape[0]
                        num_classes = out_features
                        logging.info(f"Inferred num_classes={num_classes} from state_dict key {k}")
                        break
            if num_classes is None:
                raise ValueError("num_classes is required to build architecture for state_dict loading.")
            model = build_architecture(model_name, num_classes)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            logging.info("Loaded state_dict from checkpoint['model_state_dict']")
            return model, device

        # if data itself is a state_dict
        # detect param keys typical of models
        is_state_dict = all(isinstance(v, torch.Tensor) for v in data.values())
        if is_state_dict:
            state_dict = data
            if num_classes is None:
                for k, v in state_dict.items():
                    if k.endswith(".fc.weight") or ".classifier.1.weight" in k:
                        num_classes = v.shape[0]
                        logging.info(f"Inferred num_classes={num_classes} from state_dict key {k}")
                        break
            if num_classes is None:
                raise ValueError("num_classes is required to build architecture for state_dict loading.")
            model = build_architecture(model_name, num_classes)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            logging.info("Loaded state_dict from .pth file")
            return model, device

        # else: unknown dict format (maybe contains metadata + state)
        # try common patterns
        possible_keys = ["state_dict", "model", "weights"]
        for k in possible_keys:
            if k in data and isinstance(data[k], dict):
                state_dict = data[k]
                if num_classes is None:
                    for sk, sv in state_dict.items():
                        if sk.endswith(".fc.weight") or ".classifier.1.weight" in sk:
                            num_classes = sv.shape[0]
                            logging.info(f"Inferred num_classes={num_classes} from state_dict key {sk}")
                            break
                if num_classes is None:
                    raise ValueError("num_classes is required to build architecture for checkpoint format.")
                model = build_architecture(model_name, num_classes)
                model.load_state_dict(state_dict)
                model.to(device).eval()
                logging.info(f"Loaded state_dict from .pth key '{k}'")
                return model, device

    raise ValueError("Could not interpret the .pth file format. It is not an nn.Module nor a plain state_dict. "
                     "If you saved a custom checkpoint, please provide how it was saved or re-save model.state_dict().")
