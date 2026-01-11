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
    elif model_name.startswith("efficientnet_b"):
        # support efficientnet_b0 .. efficientnet_b4 if available
        try:
            # dynamically get the constructor from torchvision.models
            ctor = getattr(models, model_name)
        except Exception:
            raise ValueError(f"Unsupported MODEL_NAME {model_name}")
        base = ctor(pretrained=False)
        # torchvision efficientnet classifier usually at classifier[1]
        try:
            in_features = base.classifier[1].in_features
            base.classifier[1] = nn.Linear(in_features, num_classes)
        except Exception:
            # fallback for other layouts
            raise ValueError(f"Unexpected efficientnet classifier layout for {model_name}")
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

    def _clean_state_dict(sd: dict):
        # remove common DataParallel 'module.' prefix if present
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
            new_sd[nk] = v
        return new_sd

    # If the checkpoint includes a list of class names, infer num_classes from it
    if isinstance(data, dict) and 'classes' in data and num_classes is None:
        try:
            cls = data['classes']
            if isinstance(cls, (list, tuple)) and len(cls) > 0:
                num_classes = len(cls)
                logging.info(f"Inferred num_classes={num_classes} from checkpoint 'classes' list")
        except Exception:
            pass

    # case 1: full model object (nn.Module)
    if isinstance(data, nn.Module):
        model = data
        model.to(device)
        model.eval()
        logging.info("Loaded full model object from .pth")
        return model, device

    # case 2: state_dict or checkpoint dict
    if isinstance(data, dict):
        # Guess whether it's a checkpoint with keys like 'model_state_dict' or 'model_state'
        if "model_state_dict" in data and isinstance(data["model_state_dict"], dict):
            state_dict = data["model_state_dict"]
        elif "model_state" in data and isinstance(data["model_state"], dict):
            state_dict = data["model_state"]
            # try to infer num_classes from final layer size if possible
            # clean keys from DataParallel wrappers
            state_dict = _clean_state_dict(state_dict)
            if num_classes is None:
                # attempt to find fc / classifier weight param and infer out_features
                for k, v in state_dict.items():
                    if k.endswith(".fc.weight") or ".classifier.1.weight" in k or k.endswith('.classifier.weight'):
                        out_features = v.shape[0]
                        num_classes = out_features
                        logging.info(f"Inferred num_classes={num_classes} from state_dict key {k}")
                        break
            if num_classes is None:
                raise ValueError("num_classes is required to build architecture for state_dict loading.")

            # Try candidate architectures: prefer specified model_name, but fall back to other common backbones
            candidates = []
            if model_name:
                candidates.append(model_name)
            # common fallbacks
            for cand in ("resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"):
                if cand not in candidates:
                    candidates.append(cand)

            last_err = None
            for cand in candidates:
                try:
                    logging.info(f"Trying architecture {cand} with num_classes={num_classes}")
                    candidate_model = build_architecture(cand, num_classes)
                    candidate_model.load_state_dict(state_dict)
                    candidate_model.to(device).eval()
                    logging.info(f"Successfully loaded state_dict using architecture {cand}")
                    return candidate_model, device
                except Exception as e:
                    last_err = e
                    logging.info(f"Failed to load state_dict with {cand}: {e}")
            # if none succeeded, raise the last error
            raise last_err

        # if data itself is a state_dict (mapping of parameter tensors)
        # detect param keys typical of models
        is_state_dict = all(isinstance(v, torch.Tensor) for v in data.values())
        if is_state_dict:
            state_dict = _clean_state_dict(data)
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
        possible_keys = ["state_dict", "model", "weights", "model_state"]
        for k in possible_keys:
            if k in data and isinstance(data[k], dict):
                state_dict = data[k]
                state_dict = _clean_state_dict(state_dict)
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
