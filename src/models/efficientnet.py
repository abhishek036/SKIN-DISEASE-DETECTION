"""
Skin Disease Detection - EfficientNet Model

EfficientNet-B0 backbone with custom classification head for skin lesion detection.
Supports transfer learning from ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, List, Tuple


class SkinLesionClassifier(nn.Module):
    """
    EfficientNet-based classifier for skin lesion detection.
    
    Architecture:
        - EfficientNet-B0 backbone (pretrained on ImageNet)
        - Global Average Pooling
        - Dropout for regularization
        - Fully connected classification head
    
    Args:
        num_classes: Number of output classes (default: 8 for ISIC 2019)
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout probability before final layer
        freeze_backbone: Freeze backbone layers initially
    """
    
    # ISIC 2019 class names
    CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    
    CLASS_FULL_NAMES = {
        'AK': 'Actinic Keratosis',
        'BCC': 'Basal Cell Carcinoma',
        'BKL': 'Benign Keratosis',
        'DF': 'Dermatofibroma',
        'MEL': 'Melanoma',
        'NV': 'Melanocytic Nevus',
        'SCC': 'Squamous Cell Carcinoma',
        'VASC': 'Vascular Lesion',
    }
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Get the number of features from backbone
        self.num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
        
        # Store feature maps for Grad-CAM
        self.gradients = None
        self.activations = None
        
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
            
    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone parameters.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreeze all layers.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        if num_layers is not None:
            # Re-freeze early layers
            layers = list(self.backbone.features.children())
            for layer in layers[:-num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            print(f"Unfroze last {num_layers} layers")
        else:
            print("All backbone layers unfrozen")
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, 224, 224)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    # =========================================================================
    # Grad-CAM Support
    # =========================================================================
    
    def _save_gradients(self, grad):
        """Hook to save gradients."""
        self.gradients = grad
    
    def _save_activations(self, module, input, output):
        """Hook to save activations."""
        self.activations = output
        output.register_hook(self._save_gradients)
    
    def register_gradcam_hooks(self) -> None:
        """Register hooks for Grad-CAM on the last convolutional layer."""
        # EfficientNet-B0: last conv layer is in features[-1]
        target_layer = self.backbone.features[-1]
        target_layer.register_forward_hook(self._save_activations)
    
    def get_gradcam(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input image tensor (1, 3, H, W)
            target_class: Target class index. If None, use predicted class.
            
        Returns:
            Heatmap tensor of shape (H, W)
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(x)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass
        self.zero_grad()
        logits[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze()


class EfficientNetB3Classifier(SkinLesionClassifier):
    """
    EfficientNet-B3 variant for higher accuracy.
    
    Larger model (12M params) with input size 300x300.
    Better accuracy at the cost of slower inference.
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        # Skip parent __init__, we override everything
        nn.Module.__init__(self)
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load EfficientNet-B3
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        
        self.num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()
        
        self.gradients = None
        self.activations = None


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3-Large for mobile/edge deployment.
    
    Optimized for inference speed with reasonable accuracy.
    ~5.4M parameters, suitable for mobile apps.
    """
    
    CLASS_NAMES = SkinLesionClassifier.CLASS_NAMES
    
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        
        self.num_features = self.backbone.classifier[0].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    model_name: str = 'efficientnet_b0',
    num_classes: int = 8,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to build model by name.
    
    Args:
        model_name: One of 'efficientnet_b0', 'efficientnet_b3', 'mobilenet_v3'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate
        freeze_backbone: Freeze backbone initially
        
    Returns:
        PyTorch model
    """
    models_dict = {
        'efficientnet_b0': SkinLesionClassifier,
        'efficientnet_b3': EfficientNetB3Classifier,
        'mobilenet_v3': MobileNetV3Classifier,
    }
    
    if model_name.lower() not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from {list(models_dict.keys())}")
    
    model_class = models_dict[model_name.lower()]
    
    return model_class(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Checkpoint dict with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        loss: Current loss
        metrics: Dict of metrics
        path: Save path
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }, path)


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")
    
    for model_name in ['efficientnet_b0', 'efficientnet_b3', 'mobilenet_v3']:
        model = build_model(model_name, num_classes=8, pretrained=False)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        
        print(f"{model_name}:")
        print(f"  Output shape: {out.shape}")
        print(f"  Total params: {model.get_num_trainable_params():,}")
        print()
