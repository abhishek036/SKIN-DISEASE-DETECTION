"""
Skin Disease Detection - Grad-CAM Explainability Module

Visual explanations for model predictions using Grad-CAM.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations highlighting regions
    important for the model's prediction.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks on target layer
        if target_layer is None:
            # Default: last layer of backbone features for EfficientNet
            target_layer = self._find_target_layer()
        
        self._register_hooks(target_layer)
    
    def _find_target_layer(self):
        """Find the last convolutional layer."""
        # For EfficientNet models
        if hasattr(self.model, 'backbone'):
            return self.model.backbone.features[-1]
        # For models with features attribute
        elif hasattr(self.model, 'features'):
            return self.model.features[-1]
        else:
            raise ValueError("Could not find target layer automatically")
    
    def _register_hooks(self, target_layer):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap, target_class, confidence


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply colormap to heatmap.
    
    Args:
        heatmap: Grayscale heatmap (H, W) in [0, 1]
        colormap: OpenCV colormap
        
    Returns:
        Colored heatmap (H, W, 3) in [0, 255]
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (H, W, 3) in [0, 255]
        heatmap: Grayscale heatmap (H, W) in [0, 1]
        alpha: Transparency of heatmap
        
    Returns:
        Overlaid image (H, W, 3) in [0, 255]
    """
    colored_heatmap = apply_colormap(heatmap)
    
    # Resize heatmap to match image if needed
    if colored_heatmap.shape[:2] != image.shape[:2]:
        colored_heatmap = cv2.resize(
            colored_heatmap,
            (image.shape[1], image.shape[0])
        )
    
    # Blend
    overlaid = (1 - alpha) * image + alpha * colored_heatmap
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
    
    return overlaid


def visualize_gradcam(
    model: torch.nn.Module,
    image_path: str,
    transform,
    class_names: List[str],
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> Tuple[int, float]:
    """
    Visualize Grad-CAM for a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        transform: Preprocessing transform
        class_names: List of class names
        device: Device
        save_path: Path to save visualization
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image)
    
    # Preprocess
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap, pred_class, confidence = gradcam(input_tensor)
    gradcam.remove_hooks()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    # Resize original to match heatmap size for overlay
    resized = cv2.resize(original_array, (heatmap.shape[1], heatmap.shape[0]))
    overlay = overlay_heatmap(resized, heatmap, alpha=0.5)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {class_names[pred_class]} ({confidence:.1%})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.close()
    
    return pred_class, confidence


def batch_gradcam(
    model: torch.nn.Module,
    image_paths: List[str],
    transform,
    class_names: List[str],
    device: str = 'cuda',
    output_dir: str = 'gradcam_outputs'
) -> None:
    """
    Generate Grad-CAM for multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        transform: Preprocessing transform
        class_names: Class names
        device: Device
        output_dir: Output directory
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gradcam = GradCAM(model)
    
    for img_path in image_paths:
        img_name = Path(img_path).stem
        
        original_image = Image.open(img_path).convert('RGB')
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        
        heatmap, pred_class, confidence = gradcam(input_tensor)
        
        # Save visualization
        visualize_gradcam(
            model, img_path, transform, class_names, device,
            save_path=str(output_dir / f'{img_name}_gradcam.png')
        )
    
    gradcam.remove_hooks()
    print(f"Generated {len(image_paths)} Grad-CAM visualizations in {output_dir}")
