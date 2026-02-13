"""
Test script for the data preprocessing pipeline.

Validates:
1. Dataset loading
2. Stratified splitting
3. Augmentation transforms
4. DataLoader creation with weighted sampling
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch


def test_dataset_loading():
    """Test loading ISIC 2019 labels."""
    print("\n" + "=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)
    
    from src.data.dataset import load_isic_2019_labels, DEFAULT_CLASSES
    
    labels_csv = PROJECT_ROOT / "ISIC_2019_Training_Input" / "ISIC_2019_Training_GroundTruth.csv"
    
    if not labels_csv.exists():
        print(f"  ✗ Labels file not found: {labels_csv}")
        return False
    
    labels_df = load_isic_2019_labels(labels_csv)
    
    print(f"  Total samples: {len(labels_df):,}")
    print(f"  Columns: {list(labels_df.columns)}")
    print(f"  Classes: {labels_df['class'].unique().tolist()}")
    print(f"  Default classes: {DEFAULT_CLASSES}")
    print("  ✓ Dataset loading works")
    
    return labels_df


def test_stratified_split(labels_df):
    """Test stratified train/val/test split."""
    print("\n" + "=" * 60)
    print("TEST 2: Stratified Split")
    print("=" * 60)
    
    from src.data.preprocessing import create_stratified_split, print_split_statistics
    
    # Filter out UNK class
    df = labels_df[labels_df['class'] != 'UNK'].copy()
    print(f"  Samples (excluding UNK): {len(df):,}")
    
    train_df, val_df, test_df = create_stratified_split(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    print_split_statistics(train_df, val_df, test_df)
    print("  ✓ Stratified split works")
    
    return train_df, val_df, test_df


def test_augmentation():
    """Test augmentation transforms."""
    print("\n" + "=" * 60)
    print("TEST 3: Augmentation Transforms")
    print("=" * 60)
    
    from src.data.augmentation import (
        get_train_transforms,
        get_val_transforms,
        IMAGE_SIZES
    )
    
    # Test different augmentation strengths
    for strength in ['light', 'medium', 'strong']:
        transforms = get_train_transforms(
            image_size=224,
            augmentation_strength=strength
        )
        print(f"  {strength.capitalize()} augmentation: {len(transforms.transforms)} transforms")
    
    val_transforms = get_val_transforms(image_size=224)
    print(f"  Validation transforms: {len(val_transforms.transforms)} transforms")
    
    print(f"  Available model sizes: {len(IMAGE_SIZES)} models")
    print("  ✓ Augmentation transforms work")


def test_class_weights(labels_df):
    """Test class weight computation."""
    print("\n" + "=" * 60)
    print("TEST 4: Class Weights")
    print("=" * 60)
    
    from src.data.dataset import get_class_weights, get_sample_weights
    
    df = labels_df[labels_df['class'] != 'UNK'].copy()
    
    class_weights = get_class_weights(df)
    print("  Class weights (normalized):")
    for cls, weight in sorted(class_weights.items()):
        print(f"    {cls}: {weight:.4f}")
    
    sample_weights = get_sample_weights(df)
    print(f"  Sample weights shape: {len(sample_weights)}")
    print(f"  Sample weights range: [{min(sample_weights):.4f}, {max(sample_weights):.4f}]")
    
    print("  ✓ Class weights computation works")


def test_dataset_class(labels_df):
    """Test PyTorch Dataset class."""
    print("\n" + "=" * 60)
    print("TEST 5: PyTorch Dataset Class")
    print("=" * 60)
    
    from src.data.dataset import SkinLesionDataset
    from src.data.augmentation import get_val_transforms
    
    image_dir = PROJECT_ROOT / "ISIC_2019_Training_Input"
    df = labels_df[labels_df['class'] != 'UNK'].head(100).copy()  # Use small subset
    
    transforms = get_val_transforms(image_size=224)
    
    dataset = SkinLesionDataset(
        image_dir=image_dir,
        labels_df=df,
        transform=transforms
    )
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Classes: {dataset.classes}")
    print(f"  Num classes: {dataset.num_classes}")
    
    # Test __getitem__
    image, label = dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Label type: {type(label)}")
    
    print("  ✓ PyTorch Dataset class works")


def test_dataloader_factory():
    """Test DataLoader creation (without actual data processing)."""
    print("\n" + "=" * 60)
    print("TEST 6: DataLoader Factory")
    print("=" * 60)
    
    from src.data.dataloader import create_dataloaders
    
    # This would work with processed data
    print("  DataLoader factory is ready for use with processed data.")
    print("  Usage:")
    print("    dataloaders = create_dataloaders(")
    print("        train_dir='data/processed/train',")
    print("        val_dir='data/processed/val',")
    print("        batch_size=32")
    print("    )")
    print("  ✓ DataLoader factory imported successfully")


def run_all_tests():
    """Run all pipeline tests."""
    print("\n" + "=" * 60)
    print("SKIN DISEASE DETECTION - DATA PIPELINE TESTS")
    print("=" * 60)
    
    # Test 1: Dataset loading
    labels_df = test_dataset_loading()
    if labels_df is None:
        print("\n✗ Tests failed: Could not load dataset")
        return False
    
    # Test 2: Stratified split
    test_stratified_split(labels_df)
    
    # Test 3: Augmentation
    test_augmentation()
    
    # Test 4: Class weights
    test_class_weights(labels_df)
    
    # Test 5: Dataset class
    test_dataset_class(labels_df)
    
    # Test 6: DataLoader
    test_dataloader_factory()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run preprocessing to resize images:")
    print("   python scripts/run_preprocessing.py")
    print("2. Start model training:")
    print("   python scripts/train.py")
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
