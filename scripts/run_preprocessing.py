"""
Skin Disease Detection - Data Preprocessing Script

Run this script to preprocess raw ISIC 2019 images:
- Stratified train/val/test split (80/10/10)
- Resize images to 224x224
- Generate labels.csv for each split

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --image-size 299  # For EfficientNet-B3+
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    prepare_processed_dataset,
    create_stratified_split,
    print_split_statistics,
    compute_dataset_statistics
)
from src.data.dataset import load_isic_2019_labels


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ISIC 2019 dataset for skin lesion classification"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "ISIC_2019_Training_Input"),
        help="Directory containing raw ISIC images"
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Path to ground truth CSV (auto-detected if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size (default: 224 for EfficientNet-B0)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute dataset mean/std statistics"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show split statistics without processing images"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        sys.exit(1)
    
    # Auto-detect labels CSV
    if args.labels_csv is None:
        labels_csv = raw_dir / "ISIC_2019_Training_GroundTruth.csv"
        if not labels_csv.exists():
            labels_csv = raw_dir.parent / "ISIC_2019_Training_GroundTruth.csv"
    else:
        labels_csv = Path(args.labels_csv)
    
    if not labels_csv.exists():
        print(f"ERROR: Labels CSV not found: {labels_csv}")
        sys.exit(1)
    
    print("=" * 60)
    print("SKIN DISEASE DETECTION - DATA PREPROCESSING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Raw data directory: {raw_dir}")
    print(f"  Labels CSV: {labels_csv}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Split ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"  Random seed: {args.seed}")
    
    # Load labels
    print("\nLoading labels...")
    labels_df = load_isic_2019_labels(labels_csv)
    labels_df = labels_df[labels_df['class'] != 'UNK']
    print(f"Total samples (excluding UNK): {len(labels_df):,}")
    
    # Show class distribution
    print("\nOriginal class distribution:")
    for cls, count in labels_df['class'].value_counts().items():
        pct = count / len(labels_df) * 100
        print(f"  {cls}: {count:,} ({pct:.1f}%)")
    
    # Create stratified split (preview)
    train_df, val_df, test_df = create_stratified_split(
        labels_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    print_split_statistics(train_df, val_df, test_df)
    
    if args.dry_run:
        print("\n[DRY RUN] No images processed. Remove --dry-run to execute.")
        return
    
    # Compute statistics if requested
    if args.compute_stats:
        print("\nComputing dataset statistics...")
        mean, std = compute_dataset_statistics(raw_dir, labels_df, sample_size=1000)
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
    
    # Run full preprocessing
    print("\nStarting preprocessing pipeline...")
    processed_splits = prepare_processed_dataset(
        raw_data_dir=raw_dir,
        processed_data_dir=args.output_dir,
        labels_csv=labels_csv,
        target_size=(args.image_size, args.image_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput saved to: {args.output_dir}")
    print(f"  train/: {len(processed_splits['train']):,} images")
    print(f"  val/:   {len(processed_splits['val']):,} images")
    print(f"  test/:  {len(processed_splits['test']):,} images")
    print("\nNext steps:")
    print("  1. Verify images in data/processed/")
    print("  2. Run training: python scripts/train.py")


if __name__ == "__main__":
    main()
