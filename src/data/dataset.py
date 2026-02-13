"""
Skin Disease Detection - PyTorch Dataset Module

This module provides Dataset classes for loading and handling
skin lesion images from ISIC 2019 dataset.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for skin lesion classification.
    
    Loads images and labels from ISIC 2019 dataset format.
    Supports custom transforms for training/validation.
    
    Args:
        data_dir: Path to directory containing images
        labels_df: DataFrame with 'image' and 'class' columns
        transform: Optional transform to apply to images
        class_to_idx: Optional mapping from class names to indices
    """
    
    # ISIC 2019 class definitions
    CLASS_NAMES = {
        'MEL': 'Melanoma',
        'NV': 'Melanocytic Nevus',
        'BCC': 'Basal Cell Carcinoma',
        'AK': 'Actinic Keratosis',
        'BKL': 'Benign Keratosis',
        'DF': 'Dermatofibroma',
        'VASC': 'Vascular Lesion',
        'SCC': 'Squamous Cell Carcinoma',
    }
    
    # Default class ordering (alphabetical, excluding UNK)
    DEFAULT_CLASSES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Filter out UNK class if present
        self.labels_df = labels_df[labels_df['class'] != 'UNK'].copy()
        self.labels_df.reset_index(drop=True, inplace=True)
        
        # Create class mapping
        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.DEFAULT_CLASSES)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Validate that all images exist
        self._validate_images()
        
    def _validate_images(self) -> None:
        """Check that all referenced images exist."""
        missing = []
        for idx, row in self.labels_df.iterrows():
            img_path = self._get_image_path(row['image'])
            if not img_path.exists():
                missing.append(row['image'])
        
        if missing:
            print(f"Warning: {len(missing)} images not found. First 5: {missing[:5]}")
            # Remove missing images from dataframe
            self.labels_df = self.labels_df[~self.labels_df['image'].isin(missing)]
            self.labels_df.reset_index(drop=True, inplace=True)
    
    def _get_image_path(self, image_id: str) -> Path:
        """Get full path to image file."""
        # Handle both with and without .jpg extension
        if not image_id.endswith('.jpg'):
            image_id = f"{image_id}.jpg"
        return self.data_dir / image_id
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            tuple: (image_tensor, label_index)
        """
        row = self.labels_df.iloc[idx]
        
        # Load image
        img_path = self._get_image_path(row['image'])
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.class_to_idx[row['class']]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        return self.labels_df['class'].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights inversely proportional to frequency.
        Useful for weighted loss functions.
        
        Returns:
            Tensor of shape (num_classes,) with class weights
        """
        counts = self.labels_df['class'].value_counts()
        total = len(self.labels_df)
        
        weights = []
        for cls in self.DEFAULT_CLASSES:
            if cls in counts:
                # Inverse frequency weighting
                weight = total / (len(counts) * counts[cls])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler.
        
        Returns:
            Tensor of shape (num_samples,) with sample weights
        """
        class_weights = self.get_class_weights()
        sample_weights = []
        
        for _, row in self.labels_df.iterrows():
            class_idx = self.class_to_idx[row['class']]
            sample_weights.append(class_weights[class_idx].item())
        
        return torch.FloatTensor(sample_weights)


class SkinLesionInferenceDataset(Dataset):
    """
    Dataset for inference (no labels required).
    
    Args:
        image_paths: List of paths to images
        transform: Transform to apply to images
    """
    
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform: Optional[Callable] = None
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single sample.
        
        Returns:
            tuple: (image_tensor, image_filename)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path.name


def load_isic_2019_labels(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and process ISIC 2019 ground truth CSV.
    
    Converts one-hot encoded labels to single class column.
    
    Args:
        csv_path: Path to ISIC_2019_Training_GroundTruth.csv
        
    Returns:
        DataFrame with 'image' and 'class' columns
    """
    df = pd.read_csv(csv_path)
    
    # Class columns (one-hot encoded)
    class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    # Convert one-hot to class label
    df['class'] = df[class_cols].idxmax(axis=1)
    
    # Keep only relevant columns
    df = df[['image', 'class']].copy()
    
    return df


if __name__ == "__main__":
    # Quick test
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "ISIC_2019_Training_Input"
    LABELS_CSV = DATA_DIR / "ISIC_2019_Training_GroundTruth.csv"
    
    if LABELS_CSV.exists():
        labels_df = load_isic_2019_labels(LABELS_CSV)
        print(f"Loaded {len(labels_df)} labels")
        print(f"Class distribution:\n{labels_df['class'].value_counts()}")
        
        # Test dataset (without transforms for now)
        dataset = SkinLesionDataset(DATA_DIR, labels_df)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Class weights: {dataset.get_class_weights()}")
    else:
        print(f"Labels file not found: {LABELS_CSV}")
