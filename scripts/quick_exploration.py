"""Quick Data Exploration - Minimal version for Python 3.14"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MPLBACKEND'] = 'Agg'

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\Admin\Downloads\SKIN")

print("=" * 60)
print("SKIN DISEASE DETECTION - QUICK DATA EXPLORATION")
print("=" * 60)

# ISIC 2019
print("\n[1] ISIC 2019 DATASET")
print("-" * 40)
isic_gt = pd.read_csv(PROJECT_ROOT / "ISIC_2019_Training_Input" / "ISIC_2019_Training_GroundTruth.csv")
print(f"Total images: {len(isic_gt):,}")

class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
isic_gt['class'] = isic_gt[class_cols].idxmax(axis=1)
class_dist = isic_gt['class'].value_counts()

CLASS_NAMES = {'MEL': 'Melanoma', 'NV': 'Nevus', 'BCC': 'Basal Cell Carcinoma', 
               'AK': 'Actinic Keratosis', 'BKL': 'Benign Keratosis', 'DF': 'Dermatofibroma',
               'VASC': 'Vascular', 'SCC': 'Squamous Cell Carcinoma', 'UNK': 'Unknown'}

print("\nClass Distribution:")
for cls, count in class_dist.items():
    print(f"  {cls:5}: {count:6,} ({count/len(isic_gt)*100:5.2f}%) - {CLASS_NAMES[cls]}")

print(f"\nImbalance ratio: {class_dist.max()/class_dist.min():.1f}:1")

# HAM10000
print("\n[2] HAM10000 DATASET")
print("-" * 40)
ham_meta = pd.read_csv(PROJECT_ROOT / "skin-cancer-mnist-ham10000" / "HAM10000_metadata.csv")
print(f"Total images: {len(ham_meta):,}")

ham_dist = ham_meta['dx'].value_counts()
HAM_NAMES = {'akiec': 'Actinic Keratosis', 'bcc': 'Basal Cell Carcinoma', 
             'bkl': 'Benign Keratosis', 'df': 'Dermatofibroma', 
             'mel': 'Melanoma', 'nv': 'Nevus', 'vasc': 'Vascular'}

print("\nClass Distribution:")
for cls, count in ham_dist.items():
    print(f"  {cls:6}: {count:6,} ({count/len(ham_meta)*100:5.2f}%) - {HAM_NAMES[cls]}")

print(f"\nImbalance ratio: {ham_dist.max()/ham_dist.min():.1f}:1")

# Overlap check
print("\n[3] DATASET OVERLAP")
print("-" * 40)
isic_ids = set(isic_gt['image'].str.replace('_downsampled', ''))
ham_ids = set(ham_meta['image_id'])
overlap = isic_ids & ham_ids
print(f"ISIC IDs: {len(isic_ids):,}")
print(f"HAM IDs: {len(ham_ids):,}")
print(f"Overlap: {len(overlap):,}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Combined dataset potential: {len(isic_gt) + len(ham_meta) - len(overlap):,} unique images

Key Issues to Address:
1. Severe class imbalance (Nevus ~50-67% of data)
2. Minority classes need augmentation (DF, VASC, SCC < 5%)
3. {len(overlap):,} duplicate images to remove
4. Variable image sizes need standardization to 224x224

Next Step: Phase 2 - Data Preprocessing Pipeline
""")

# Save summary
summary_df = pd.DataFrame({
    'Dataset': ['ISIC 2019', 'HAM10000'],
    'Images': [len(isic_gt), len(ham_meta)],
    'Classes': [9, 7],
    'Majority': ['NV', 'nv'],
    'Majority_Count': [class_dist.max(), ham_dist.max()],
    'Minority': [class_dist.idxmin(), ham_dist.idxmin()],
    'Minority_Count': [class_dist.min(), ham_dist.min()]
})
summary_df.to_csv(PROJECT_ROOT / 'notebooks' / 'exploration_summary.csv', index=False)
print("✓ Saved: notebooks/exploration_summary.csv")
