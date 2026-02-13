"""
Skin Disease Detection - Data Augmentation Module

Training-time augmentations using torchvision transforms.
Designed to improve model generalization for dermoscopic images.
"""

from typing import Tuple, Optional, Dict, Any
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


# Standard image sizes for different models
IMAGE_SIZES = {
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'efficientnet_b5': 456,
    'efficientnet_b6': 528,
    'efficientnet_b7': 600,
    'resnet50': 224,
    'vit_b_16': 224,
    'mobilenet_v3': 224,
}

# Normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
    augmentation_strength: str = 'medium'
) -> T.Compose:
    """
    Get training transforms with augmentation.
    
    Augmentations suitable for dermoscopic images:
    - Random rotation (lesions can appear at any angle)
    - Horizontal/vertical flip (orientation doesn't matter)
    - Color jitter (account for lighting variations)
    - Random resized crop (simulate different zoom levels)
    - Random erasing (occlusion robustness)
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        augmentation_strength: 'light', 'medium', or 'strong'
        
    Returns:
        Composed transforms
    """
    aug_params = _get_augmentation_params(augmentation_strength)
    
    transforms_list = [
        # Resize with slight random crop
        T.RandomResizedCrop(
            image_size,
            scale=(aug_params['crop_scale_min'], 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR
        ),
        
        # Geometric augmentations
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(
            degrees=aug_params['rotation_degrees'],
            interpolation=InterpolationMode.BILINEAR
        ),
        
        # Color augmentations
        T.ColorJitter(
            brightness=aug_params['brightness'],
            contrast=aug_params['contrast'],
            saturation=aug_params['saturation'],
            hue=aug_params['hue']
        ),
        
        # Random affine for slight perspective changes
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR
        ),
        
        # Convert to tensor
        T.ToTensor(),
        
        # Normalize
        T.Normalize(mean=mean, std=std),
    ]
    
    # Add random erasing for stronger augmentation
    if augmentation_strength in ['medium', 'strong']:
        transforms_list.append(
            T.RandomErasing(
                p=aug_params['erasing_prob'],
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3)
            )
        )
    
    return T.Compose(transforms_list)


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
) -> T.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
) -> T.Compose:
    """
    Get transforms for inference on new images.
    
    Same as validation transforms.
    """
    return get_val_transforms(image_size, mean, std)


def get_tta_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
    num_augmentations: int = 5
) -> list:
    """
    Get Test-Time Augmentation (TTA) transforms.
    
    Returns a list of transform compositions for TTA.
    Final prediction is averaged across all augmented versions.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        num_augmentations: Number of augmented versions
        
    Returns:
        List of composed transforms
    """
    base_transform = get_val_transforms(image_size, mean, std)
    
    tta_transforms = [base_transform]  # Original
    
    # Horizontal flip
    tta_transforms.append(T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]))
    
    # Vertical flip
    tta_transforms.append(T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]))
    
    # 90-degree rotations
    for angle in [90, 180, 270]:
        if len(tta_transforms) >= num_augmentations:
            break
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            T.functional.rotate.__call__ if hasattr(T.functional, 'rotate') else T.RandomRotation((angle, angle)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]))
    
    return tta_transforms[:num_augmentations]


def _get_augmentation_params(strength: str) -> Dict[str, Any]:
    """Get augmentation parameters based on strength level."""
    params = {
        'light': {
            'crop_scale_min': 0.9,
            'rotation_degrees': 15,
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.02,
            'erasing_prob': 0.0,
        },
        'medium': {
            'crop_scale_min': 0.8,
            'rotation_degrees': 45,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.05,
            'erasing_prob': 0.2,
        },
        'strong': {
            'crop_scale_min': 0.7,
            'rotation_degrees': 90,
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1,
            'erasing_prob': 0.4,
        },
    }
    
    if strength not in params:
        raise ValueError(f"Unknown augmentation strength: {strength}. "
                        f"Choose from {list(params.keys())}")
    
    return params[strength]


def get_transforms_for_model(
    model_name: str,
    split: str = 'train',
    augmentation_strength: str = 'medium'
) -> T.Compose:
    """
    Get appropriate transforms for a specific model and split.
    
    Args:
        model_name: Name of the model (e.g., 'efficientnet_b4')
        split: 'train', 'val', or 'test'
        augmentation_strength: Augmentation level for training
        
    Returns:
        Composed transforms
    """
    # Get image size for model
    model_name_lower = model_name.lower()
    image_size = IMAGE_SIZES.get(model_name_lower, 224)
    
    if split == 'train':
        return get_train_transforms(
            image_size=image_size,
            augmentation_strength=augmentation_strength
        )
    else:
        return get_val_transforms(image_size=image_size)


def denormalize(
    tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized tensor
    """
    import torch
    
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std + mean


def visualize_augmentations(
    image_path: str,
    transforms: T.Compose,
    num_samples: int = 6,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize augmentations applied to an image.
    
    Args:
        image_path: Path to input image
        transforms: Transforms to visualize
        num_samples: Number of augmented samples to show
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    img = Image.open(image_path).convert('RGB')
    
    fig, axes = plt.subplots(2, (num_samples + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes[:num_samples]):
        augmented = transforms(img)
        
        # Denormalize for visualization
        augmented = denormalize(augmented)
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = augmented.clip(0, 1)
        
        ax.imshow(augmented)
        ax.set_title(f'Augmentation {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150)
    plt.close()
    print("Saved augmentation samples to augmentation_samples.png")


if __name__ == "__main__":
    # Print available transforms info
    print("Available augmentation strengths: light, medium, strong")
    print("\nImage sizes by model:")
    for model, size in IMAGE_SIZES.items():
        print(f"  {model}: {size}x{size}")
