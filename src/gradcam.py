from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def gradcam_overlay(model, input_tensor, original_image: Image.Image, device):
    activations = {}
    gradients = {}
    target_layer = model.layer4[-1]

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor.to(device)).squeeze()
        logits.backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        rgb = np.array(original_image.convert("RGB"))
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.58 * rgb + 0.42 * heatmap, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    finally:
        forward_handle.remove()
        backward_handle.remove()
