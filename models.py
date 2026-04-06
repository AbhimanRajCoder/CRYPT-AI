"""
models.py — Model definitions for Off-Road Segmentation.

ROUND 3 UPGRADES:
  - ResNet101 backbone (deeper features for similar class discrimination)
  - Auxiliary classifier ENABLED (improves gradient flow to early layers)
  - Optional torch.compile() for PyTorch 2.x
"""

import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)

from config import NUM_CLASSES, BACKBONE, PRETRAINED


def build_deeplabv3(
    num_classes=NUM_CLASSES,
    backbone=BACKBONE,
    pretrained=PRETRAINED,
    use_aux=True,
):
    """
    Build a DeepLabV3 model with pretrained backbone.
    
    ROUND 3: Auxiliary classifier ENABLED for better gradient flow.
    This helps early layers learn better features for rare classes.
    """
    if backbone == "resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet101(weights=weights)
        print(f"[INFO] Loaded DeepLabV3 with ResNet101 backbone (pretrained={pretrained}, aux={use_aux})")
    elif backbone == "resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights)
        print(f"[INFO] Loaded DeepLabV3 with ResNet50 backbone (pretrained={pretrained}, aux={use_aux})")
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50' or 'resnet101'.")

    # Replace the main classifier head for our number of classes
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    # Replace aux classifier head too (if enabled)
    if use_aux and model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        print(f"[INFO] Auxiliary classifier enabled (helps gradient flow to early layers)")
    else:
        # Disable auxiliary classifier
        model.aux_classifier = None

    return model


class SegmentationModel(nn.Module):
    """
    Wrapper around DeepLabV3 for cleaner interface.
    
    Returns:
      - Training: tuple (main_output, aux_output) if aux is enabled
      - Eval: main_output only
    """

    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=PRETRAINED, use_aux=False):
        super().__init__()
        self.use_aux = use_aux
        self.model = build_deeplabv3(num_classes, backbone, pretrained, use_aux=use_aux)

    def forward(self, x):
        output = self.model(x)
        main_out = output["out"]  # (B, C, H, W)
        
        if self.training and self.use_aux and "aux" in output:
            aux_out = output["aux"]  # (B, C, H, W)
            return main_out, aux_out
        
        return main_out


# ============================================================================
# Model Summary
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters:     {total:,}")
    print(f"[INFO] Trainable parameters: {trainable:,}")
    return trainable


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    model = SegmentationModel()
    count_parameters(model)

    # Test forward pass (training mode — returns tuple)
    model.train()
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, tuple):
        print(f"Input:  {dummy.shape}")
        print(f"Main output: {out[0].shape}")
        print(f"Aux output:  {out[1].shape}")
    else:
        print(f"Input:  {dummy.shape}")
        print(f"Output: {out.shape}")

    # Test eval mode
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Eval output: {out.shape}")
