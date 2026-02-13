"""
Data Exploration Script for Skin Disease Detection
Run this instead of the notebook due to Python 3.14 kernel issues.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(r"C:\Users\Admin\Downloads\SKIN")
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("=" * 60)
print("SKIN DISEASE DETECTION - DATA EXPLORATION")
print("=" * 60)
print(f"\nPython: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")

# ============================================================
# Define Dataset Paths
# ============================================================
ISIC_2019_DIR = PROJECT_ROOT / "ISIC_2019_Training_Input"
ISIC_2019_GT = ISIC_2019_DIR / "ISIC_2019_Training_GroundTruth.csv"
ISIC_2019_META = ISIC_2019_DIR / "ISIC_2019_Training_Metadata.csv"

HAM10000_DIR = PROJECT_ROOT / "skin-cancer-mnist-ham10000"
HAM10000_META = HAM10000_DIR / "HAM10000_metadata.csv"
HAM10000_IMAGES_1 = HAM10000_DIR / "HAM10000_images_part_1"
HAM10000_IMAGES_2 = HAM10000_DIR / "HAM10000_images_part_2"

OUTPUT_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR.mkdir(exist_ok=True)

# Verify paths
print("\n" + "=" * 60)
print("DATASET AVAILABILITY CHECK")
print("=" * 60)
datasets = {
    "ISIC 2019 Images": ISIC_2019_DIR,
    "ISIC 2019 Ground Truth": ISIC_2019_GT,
    "HAM10000 Metadata": HAM10000_META,
    "HAM10000 Images Part 1": HAM10000_IMAGES_1,
    "HAM10000 Images Part 2": HAM10000_IMAGES_2
}

for name, path in datasets.items():
    status = "✓ Found" if path.exists() else "✗ Missing"
    print(f"  {name}: {status}")

# ============================================================
# ISIC 2019 Analysis
# ============================================================
print("\n" + "=" * 60)
print("1. ISIC 2019 DATASET ANALYSIS")
print("=" * 60)

ISIC_CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'BCC': 'Basal Cell Carcinoma',
    'AK': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesion',
    'SCC': 'Squamous Cell Carcinoma',
    'UNK': 'Unknown'
}

isic_gt = pd.read_csv(ISIC_2019_GT)
print(f"\nISIC 2019 Ground Truth Shape: {isic_gt.shape}")

# Convert one-hot to class labels
class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
isic_gt['class'] = isic_gt[class_cols].idxmax(axis=1)

print("\nISIC 2019 Class Distribution:")
print("-" * 50)
class_dist = isic_gt['class'].value_counts()
for cls, count in class_dist.items():
    pct = (count / len(isic_gt)) * 100
    print(f"  {cls:6} ({ISIC_CLASS_NAMES[cls]:25}): {count:6,} ({pct:5.2f}%)")

print(f"\n⚠️  Class Imbalance:")
print(f"    Majority class (NV): {class_dist.max():,} samples")
print(f"    Minority class ({class_dist.idxmin()}): {class_dist.min():,} samples")
print(f"    Imbalance ratio: {class_dist.max() / class_dist.min():.1f}:1")

# Plot ISIC 2019 distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = sns.color_palette('husl', len(class_dist))

bars = axes[0].bar(class_dist.index, class_dist.values, color=colors)
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('ISIC 2019 Class Distribution', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
for bar, count in zip(bars, class_dist.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

axes[1].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('ISIC 2019 Class Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'isic_2019_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: isic_2019_class_distribution.png")

# ============================================================
# HAM10000 Analysis
# ============================================================
print("\n" + "=" * 60)
print("2. HAM10000 DATASET ANALYSIS")
print("=" * 60)

HAM_CLASS_NAMES = {
    'akiec': "Actinic Keratoses / Bowen's Disease",
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

ham_meta = pd.read_csv(HAM10000_META)
print(f"\nHAM10000 Metadata Shape: {ham_meta.shape}")

print("\nHAM10000 Class Distribution:")
print("-" * 50)
ham_class_dist = ham_meta['dx'].value_counts()
for cls, count in ham_class_dist.items():
    pct = (count / len(ham_meta)) * 100
    print(f"  {cls:6} ({HAM_CLASS_NAMES[cls]:35}): {count:6,} ({pct:5.2f}%)")

print(f"\n⚠️  Class Imbalance:")
print(f"    Majority class (nv): {ham_class_dist.max():,} samples")
print(f"    Minority class ({ham_class_dist.idxmin()}): {ham_class_dist.min():,} samples")
print(f"    Imbalance ratio: {ham_class_dist.max() / ham_class_dist.min():.1f}:1")

# Plot HAM10000 distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = sns.color_palette('husl', len(ham_class_dist))

bars = axes[0].bar(ham_class_dist.index, ham_class_dist.values, color=colors)
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('HAM10000 Class Distribution', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
for bar, count in zip(bars, ham_class_dist.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

axes[1].pie(ham_class_dist.values, labels=ham_class_dist.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('HAM10000 Class Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ham10000_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: ham10000_class_distribution.png")

# HAM10000 Demographics
print("\nHAM10000 Demographics:")
print("-" * 50)
print(f"  Age range: {ham_meta['age'].min():.0f} - {ham_meta['age'].max():.0f} years")
print(f"  Mean age: {ham_meta['age'].mean():.1f} years")
print(f"  Missing age: {ham_meta['age'].isna().sum()} records")
print(f"\n  Sex distribution:")
for sex, count in ham_meta['sex'].value_counts().items():
    print(f"    {sex}: {count:,} ({count/len(ham_meta)*100:.1f}%)")

# Demographics plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ham_meta['age'].hist(bins=30, ax=axes[0, 0], color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Age', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].set_title('Age Distribution', fontsize=14, fontweight='bold')
axes[0, 0].axvline(ham_meta['age'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {ham_meta['age'].mean():.1f}")
axes[0, 0].legend()

sex_dist = ham_meta['sex'].value_counts()
axes[0, 1].pie(sex_dist.values, labels=sex_dist.index, autopct='%1.1f%%',
               colors=['#3498db', '#e74c3c'], startangle=90)
axes[0, 1].set_title('Sex Distribution', fontsize=14, fontweight='bold')

loc_dist = ham_meta['localization'].value_counts().head(10)
axes[1, 0].barh(loc_dist.index, loc_dist.values, color='teal')
axes[1, 0].set_xlabel('Count', fontsize=12)
axes[1, 0].set_ylabel('Body Location', fontsize=12)
axes[1, 0].set_title('Top 10 Lesion Localizations', fontsize=14, fontweight='bold')

dx_type_dist = ham_meta['dx_type'].value_counts()
axes[1, 1].bar(dx_type_dist.index, dx_type_dist.values, color='coral')
axes[1, 1].set_xlabel('Diagnosis Type', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
axes[1, 1].set_title('Diagnosis Method', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ham10000_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: ham10000_demographics.png")

# ============================================================
# Image Properties Analysis
# ============================================================
print("\n" + "=" * 60)
print("3. IMAGE PROPERTIES ANALYSIS")
print("=" * 60)

def analyze_images(image_dir, sample_size=200):
    """Analyze image properties from a directory."""
    image_files = list(Path(image_dir).glob('*.jpg'))
    
    if len(image_files) > sample_size:
        np.random.seed(42)
        image_files = list(np.random.choice(image_files, sample_size, replace=False))
    
    widths, heights, file_sizes = [], [], []
    corrupt_images = []
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                file_sizes.append(os.path.getsize(img_path) / 1024)
        except Exception as e:
            corrupt_images.append(img_path.name)
    
    return {
        'total': len(image_files),
        'widths': widths,
        'heights': heights,
        'file_sizes': file_sizes,
        'corrupt': corrupt_images,
        'unique_dimensions': len(set(zip(widths, heights)))
    }

print("\nAnalyzing ISIC 2019 images (sample of 200)...")
isic_stats = analyze_images(ISIC_2019_DIR, sample_size=200)

print(f"\nISIC 2019 Image Statistics:")
print(f"  Sampled: {isic_stats['total']} images")
print(f"  Width range: {min(isic_stats['widths'])} - {max(isic_stats['widths'])} px")
print(f"  Height range: {min(isic_stats['heights'])} - {max(isic_stats['heights'])} px")
print(f"  Mean dimensions: {np.mean(isic_stats['widths']):.0f} x {np.mean(isic_stats['heights']):.0f} px")
print(f"  Unique dimensions: {isic_stats['unique_dimensions']}")
print(f"  File size range: {min(isic_stats['file_sizes']):.1f} - {max(isic_stats['file_sizes']):.1f} KB")
print(f"  Mean file size: {np.mean(isic_stats['file_sizes']):.1f} KB")
print(f"  Corrupt images: {len(isic_stats['corrupt'])}")

# HAM10000 images
print("\nAnalyzing HAM10000 images (sample of 200)...")
ham_images = []
for part_dir in [HAM10000_IMAGES_1, HAM10000_IMAGES_2]:
    if part_dir.exists():
        ham_images.extend(list(part_dir.glob('*.jpg')))

print(f"  Total HAM10000 images found: {len(ham_images)}")

if len(ham_images) > 0:
    np.random.seed(42)
    sample_images = list(np.random.choice(ham_images, min(200, len(ham_images)), replace=False))
    
    widths, heights, file_sizes = [], [], []
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                file_sizes.append(os.path.getsize(img_path) / 1024)
        except:
            pass
    
    print(f"\nHAM10000 Image Statistics:")
    print(f"  Sampled: {len(widths)} images")
    print(f"  Width range: {min(widths)} - {max(widths)} px")
    print(f"  Height range: {min(heights)} - {max(heights)} px")
    print(f"  Mean dimensions: {np.mean(widths):.0f} x {np.mean(heights):.0f} px")
    print(f"  Unique dimensions: {len(set(zip(widths, heights)))}")
    print(f"  File size range: {min(file_sizes):.1f} - {max(file_sizes):.1f} KB")
    print(f"  Mean file size: {np.mean(file_sizes):.1f} KB")

# Image dimensions plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(isic_stats['widths'], isic_stats['heights'], alpha=0.5, s=10)
axes[0].set_xlabel('Width (px)', fontsize=12)
axes[0].set_ylabel('Height (px)', fontsize=12)
axes[0].set_title('ISIC 2019 Image Dimensions', fontsize=14, fontweight='bold')
axes[0].axhline(y=224, color='r', linestyle='--', alpha=0.7, label='Target: 224px')
axes[0].axvline(x=224, color='r', linestyle='--', alpha=0.7)
axes[0].legend()

axes[1].hist(isic_stats['file_sizes'], bins=50, color='steelblue', edgecolor='black')
axes[1].set_xlabel('File Size (KB)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('ISIC 2019 File Size Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'image_dimensions_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: image_dimensions_analysis.png")

# ============================================================
# Dataset Overlap Analysis
# ============================================================
print("\n" + "=" * 60)
print("4. DATASET OVERLAP ANALYSIS")
print("=" * 60)

isic_ids = set(isic_gt['image'].str.replace('_downsampled', ''))
ham_ids = set(ham_meta['image_id'])
overlap = isic_ids.intersection(ham_ids)

print(f"\n  ISIC 2019 unique IDs: {len(isic_ids):,}")
print(f"  HAM10000 unique IDs: {len(ham_ids):,}")
print(f"  Overlapping IDs: {len(overlap):,}")
print(f"\n  Combined unique images: ~{len(isic_ids) + len(ham_ids) - len(overlap):,}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("5. SUMMARY & RECOMMENDATIONS")
print("=" * 60)

summary = """
╔══════════════════════════════════════════════════════════════════╗
║              DATA EXPLORATION COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  DATASETS:                                                       ║
║  • ISIC 2019: {:,} images, 9 classes                           ║
║  • HAM10000:  {:,} images, 7 classes                           ║
║                                                                  ║
║  KEY FINDINGS:                                                   ║
║  ✓ Severe class imbalance in both datasets                       ║
║  ✓ Nevus (NV/nv) dominates (~50-67%)                            ║
║  ✓ {:,} overlapping images between datasets                    ║
║  ✓ Variable image dimensions in ISIC 2019                        ║
║                                                                  ║
║  PREPROCESSING NEEDED:                                           ║
║  1. Resize to 224x224 for EfficientNet                          ║
║  2. Handle class imbalance (weighted loss, augmentation)         ║
║  3. Deduplicate overlapping images                               ║
║  4. Stratified train/val/test split                              ║
║                                                                  ║
║  OUTPUTS SAVED TO: notebooks/                                    ║
║  • isic_2019_class_distribution.png                              ║
║  • ham10000_class_distribution.png                               ║
║  • ham10000_demographics.png                                     ║
║  • image_dimensions_analysis.png                                 ║
║  • exploration_summary.csv                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""".format(len(isic_gt), len(ham_meta), len(overlap))

print(summary)

# Save summary CSV
exploration_results = {
    'Dataset': ['ISIC 2019', 'HAM10000'],
    'Total Images': [len(isic_gt), len(ham_meta)],
    'Classes': [9, 7],
    'Majority Class': ['NV', 'nv'],
    'Majority Count': [class_dist.max(), ham_class_dist.max()],
    'Minority Class': [class_dist.idxmin(), ham_class_dist.idxmin()],
    'Minority Count': [class_dist.min(), ham_class_dist.min()],
    'Imbalance Ratio': [f"{class_dist.max() / class_dist.min():.1f}:1", 
                        f"{ham_class_dist.max() / ham_class_dist.min():.1f}:1"]
}

results_df = pd.DataFrame(exploration_results)
results_df.to_csv(OUTPUT_DIR / 'exploration_summary.csv', index=False)
print(f"✓ Saved: exploration_summary.csv")

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETE!")
print("=" * 60)
