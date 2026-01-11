"""
Simple Grad-CAM implementation for PyTorch models.
Usage:
    from src.gradcam import GradCAM
    gcam = GradCAM(model, target_layer)
    cam, class_idx = gcam(input_tensor)
    # cam is HxW numpy floats in [0,1]
"""

import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # Register hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: 1xCxHxW on same device as model
        returns (cam, class_idx)
        - cam: numpy HxW in [0,1]
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).cpu().item())
        loss = out[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grads = self.gradients[0]  # CxHxW
        acts = self.activations[0]  # CxHxW
        weights = grads.mean(dim=(1,2))  # C
        cam = (weights[:, None, None] * acts).sum(dim=0)  # HxW
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        return cam_np, class_idx

def apply_colormap_on_image(org_im, activation_map, colormap=cv2.COLORMAP_JET, alpha=0.4):
    """
    org_im: numpy HxW3 RGB (0..255)
    activation_map: HxW floats in 0..1
    returns overlay RGB uint8
    """
    heatmap = (activation_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(org_im.astype(np.float32), 1 - alpha, heatmap.astype(np.float32), alpha, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
