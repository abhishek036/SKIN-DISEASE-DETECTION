"""
Skin Disease Detection - DataLoader Factory Module

Creates PyTorch DataLoaders with proper sampling strategies
for handling class imbalance in skin lesion classification.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import SkinLesionDataset, SkinLesionInferenceDataset
from .augmentation import get_train_transforms, get_val_transforms


def create_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    test_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augmentation_strength: str = 'medium',
    use_weighted_sampling: bool = True,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        train_dir: Directory with training images and labels.csv
        val_dir: Directory with validation images and labels.csv
        test_dir: Directory with test images and labels.csv (optional)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Target image size
        augmentation_strength: 'light', 'medium', or 'strong'
        use_weighted_sampling: Use WeightedRandomSampler for class imbalance
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Dict with 'train', 'val', and optionally 'test' DataLoaders
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    
    # Load labels
    train_labels = pd.read_csv(train_dir / 'labels.csv')
    val_labels = pd.read_csv(val_dir / 'labels.csv')
    
    # Get transforms
    train_transforms = get_train_transforms(
        image_size=image_size,
        augmentation_strength=augmentation_strength
    )
    val_transforms = get_val_transforms(image_size=image_size)
    
    # Create datasets
    train_dataset = SkinLesionDataset(
        data_dir=train_dir,
        labels_df=train_labels,
        transform=train_transforms
    )
    
    val_dataset = SkinLesionDataset(
        data_dir=val_dir,
        labels_df=val_labels,
        transform=val_transforms
    )
    
    # Create sampler for training
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop incomplete batches for stable BN
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    # Add test loader if provided
    if test_dir is not None:
        test_dir = Path(test_dir)
        test_labels = pd.read_csv(test_dir / 'labels.csv')
        
        test_dataset = SkinLesionDataset(
            data_dir=test_dir,
            labels_df=test_labels,
            transform=val_transforms
        )
        
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return dataloaders


def create_inference_dataloader(
    image_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for inference on new images.
    
    Args:
        image_dir: Directory containing images to predict
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Target image size
        pin_memory: Pin memory for GPU
        
    Returns:
        DataLoader for inference
    """
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    transforms = get_val_transforms(image_size=image_size)
    
    dataset = SkinLesionInferenceDataset(
        image_paths=image_paths,
        transform=transforms
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_class_info(train_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get class information from training directory.
    
    Args:
        train_dir: Training directory with labels.csv
        
    Returns:
        Dict with class names, weights, and counts
    """
    train_dir = Path(train_dir)
    labels_df = pd.read_csv(train_dir / 'labels.csv')
    
    class_counts = labels_df['class'].value_counts()
    classes = sorted(class_counts.index.tolist())
    
    # Compute class weights (inverse frequency)
    total = len(labels_df)
    weights = {cls: total / count for cls, count in class_counts.items()}
    
    # Normalize weights
    max_weight = max(weights.values())
    weights = {cls: w / max_weight for cls, w in weights.items()}
    
    return {
        'classes': classes,
        'num_classes': len(classes),
        'class_counts': class_counts.to_dict(),
        'class_weights': weights,
        'class_to_idx': {cls: i for i, cls in enumerate(classes)}
    }


def verify_dataloaders(dataloaders: Dict[str, DataLoader]) -> None:
    """
    Verify DataLoaders are working correctly.
    
    Args:
        dataloaders: Dict of DataLoaders to verify
    """
    print("\n" + "=" * 60)
    print("DATALOADER VERIFICATION")
    print("=" * 60)
    
    for name, loader in dataloaders.items():
        print(f"\n{name.upper()} DataLoader:")
        print(f"  Dataset size: {len(loader.dataset):,}")
        print(f"  Batch size: {loader.batch_size}")
        print(f"  Num batches: {len(loader):,}")
        print(f"  Num workers: {loader.num_workers}")
        
        # Test loading one batch
        try:
            batch = next(iter(loader))
            images, labels = batch
            print(f"  Batch shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Image dtype: {images.dtype}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  ✓ DataLoader working correctly")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def compute_effective_samples(
    dataloader: DataLoader,
    num_batches: int = 100
) -> Dict[str, int]:
    """
    Compute effective class distribution after weighted sampling.
    
    Args:
        dataloader: Training DataLoader with weighted sampling
        num_batches: Number of batches to sample
        
    Returns:
        Dict with class counts in sampled batches
    """
    from collections import Counter
    
    class_counts = Counter()
    
    for i, (_, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        class_counts.update(labels.numpy().tolist())
    
    return dict(class_counts)


if __name__ == "__main__":
    # Example usage
    print("DataLoader Factory Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from src.data.dataloader import create_dataloaders")
    print("  ")
    print("  dataloaders = create_dataloaders(")
    print("      train_dir='data/processed/train',")
    print("      val_dir='data/processed/val',")
    print("      test_dir='data/processed/test',")
    print("      batch_size=32,")
    print("      use_weighted_sampling=True")
    print("  )")
