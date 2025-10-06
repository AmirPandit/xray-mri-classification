import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Compute weights and CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
def apply_gradcam(model, input_tensor, original_image, target_class=None):
    # Ensure model is in eval mode
    model.eval()

    # Try common architectures first
    if hasattr(model, 'features'):
        target_layer = model.features[-1]
    elif hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    else:
        # Fallback: find last conv layer recursively
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise ValueError("No Conv2d layers found for Grad-CAM")
        target_layer = conv_layers[-1]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, target_class)
    
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    
    return overlayed
