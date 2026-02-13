"""
Skin Disease Detection - Data Preprocessing Module

Handles image preprocessing, train/val/test splitting,
and data preparation for model training.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Skin lesion specific statistics (can be computed from dataset)
SKIN_LESION_MEAN = [0.7635, 0.5461, 0.5705]
SKIN_LESION_STD = [0.1409, 0.1520, 0.1695]


def create_stratified_split(
    labels_df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    min_samples_per_class: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test split.
    
    Ensures each class has proportional representation in each split.
    
    Args:
        labels_df: DataFrame with 'image' and 'class' columns
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        min_samples_per_class: Minimum samples required per class
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Filter out classes with too few samples
    class_counts = labels_df['class'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = labels_df[labels_df['class'].isin(valid_classes)].copy()
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['class'],
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df['class'],
        random_state=random_state
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df


def compute_dataset_statistics(
    image_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    sample_size: int = 1000
) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std of dataset for normalization.
    
    Args:
        image_dir: Directory containing images
        labels_df: DataFrame with image IDs
        sample_size: Number of images to sample for computation
        
    Returns:
        Tuple of (mean, std) as lists of 3 floats (RGB channels)
    """
    image_dir = Path(image_dir)
    
    # Sample images
    if len(labels_df) > sample_size:
        sample_df = labels_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = labels_df
    
    # Accumulate pixel values
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Computing stats"):
        img_path = image_dir / f"{row['image']}.jpg"
        if not img_path.exists():
            continue
            
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        pixel_sum += img_array.mean(axis=(0, 1))
        pixel_sq_sum += (img_array ** 2).mean(axis=(0, 1))
        pixel_count += 1
    
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    
    return mean.tolist(), std.tolist()


def resize_and_save_images(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    target_size: Tuple[int, int] = (224, 224),
    quality: int = 95
) -> pd.DataFrame:
    """
    Resize images and save to new directory.
    
    Args:
        src_dir: Source directory with original images
        dst_dir: Destination directory for resized images
        labels_df: DataFrame with image IDs to process
        target_size: Target (width, height)
        quality: JPEG quality (1-100)
        
    Returns:
        Updated DataFrame with valid images only
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    valid_images = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Resizing"):
        image_id = row['image']
        src_path = src_dir / f"{image_id}.jpg"
        dst_path = dst_dir / f"{image_id}.jpg"
        
        if not src_path.exists():
            continue
        
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(dst_path, 'JPEG', quality=quality)
                valid_images.append(row)
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue
    
    return pd.DataFrame(valid_images).reset_index(drop=True)


def prepare_processed_dataset(
    raw_data_dir: Union[str, Path],
    processed_data_dir: Union[str, Path],
    labels_csv: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Full preprocessing pipeline: load labels, split data, resize images.
    
    Args:
        raw_data_dir: Directory with original ISIC images
        processed_data_dir: Output directory for processed data
        labels_csv: Path to ground truth CSV
        target_size: Target image size
        train_ratio, val_ratio, test_ratio: Split ratios
        random_state: Random seed
        
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    from .dataset import load_isic_2019_labels
    
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (processed_data_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print("Loading labels...")
    labels_df = load_isic_2019_labels(labels_csv)
    
    # Filter out UNK class
    labels_df = labels_df[labels_df['class'] != 'UNK']
    print(f"Total samples (excluding UNK): {len(labels_df)}")
    
    # Create stratified split
    print("Creating stratified split...")
    train_df, val_df, test_df = create_stratified_split(
        labels_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Resize and save images for each split
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    processed_splits = {}
    
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} set...")
        dst_dir = processed_data_dir / split_name
        
        processed_df = resize_and_save_images(
            src_dir=raw_data_dir,
            dst_dir=dst_dir,
            labels_df=split_df,
            target_size=target_size
        )
        
        # Save labels CSV
        processed_df.to_csv(dst_dir / 'labels.csv', index=False)
        processed_splits[split_name] = processed_df
        
        # Print class distribution
        print(f"  {split_name} class distribution:")
        for cls, count in processed_df['class'].value_counts().items():
            print(f"    {cls}: {count}")
    
    print("\n✓ Preprocessing complete!")
    return processed_splits


def get_class_distribution(labels_df: pd.DataFrame) -> pd.Series:
    """Get class distribution from labels DataFrame."""
    return labels_df['class'].value_counts()


def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """Print detailed statistics for each split."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT STATISTICS")
    print("=" * 60)
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\nTotal samples: {total:,}")
    print(f"  Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
    
    print("\nClass distribution by split:")
    print("-" * 60)
    
    all_classes = sorted(set(train_df['class'].unique()) | 
                        set(val_df['class'].unique()) | 
                        set(test_df['class'].unique()))
    
    print(f"{'Class':<8} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("-" * 60)
    
    for cls in all_classes:
        train_count = len(train_df[train_df['class'] == cls])
        val_count = len(val_df[val_df['class'] == cls])
        test_count = len(test_df[test_df['class'] == cls])
        total_count = train_count + val_count + test_count
        print(f"{cls:<8} {train_count:>10,} {val_count:>10,} {test_count:>10,} {total_count:>10,}")


if __name__ == "__main__":
    # Test preprocessing
    from pathlib import Path
    from .dataset import load_isic_2019_labels
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DIR = PROJECT_ROOT / "ISIC_2019_Training_Input"
    LABELS_CSV = RAW_DIR / "ISIC_2019_Training_GroundTruth.csv"
    
    if LABELS_CSV.exists():
        labels_df = load_isic_2019_labels(LABELS_CSV)
        labels_df = labels_df[labels_df['class'] != 'UNK']
        
        train_df, val_df, test_df = create_stratified_split(labels_df)
        print_split_statistics(train_df, val_df, test_df)
