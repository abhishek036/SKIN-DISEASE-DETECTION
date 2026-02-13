# Category 1 Dataset Acquisition Guide
## Everyday Skin Issues - Dataset Sources

**Created:** February 2026  
**Purpose:** Acquire training data for Category 1 conditions

---

## 🎯 PRIORITY DATASETS (Download First)

### 1. DermNet Dataset (Kaggle) - HIGHLY RECOMMENDED
```
URL: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
Size: ~19,500 images across 23 categories
License: Research use

Relevant Categories for Category 1:
├── Acne and Rosacea Photos (1,200+ images)
├── Atopic Dermatitis Photos
├── Eczema Photos
├── Nail Fungus and other Nail Disease
├── Scars Photos
├── Seborrheic Keratoses
├── Psoriasis pictures Lichen Planus
├── Warts Molluscum and other Viral Infections
└── Hair Loss Photos Alopecia
```

**Download Command (requires Kaggle API):**
```powershell
# Install Kaggle API first
pip install kaggle

# Set up credentials (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d shubhamgoel27/dermnet -p ./data/dermnet
```

---

### 2. Fitzpatrick17k Dataset - ESSENTIAL FOR DIVERSITY
```
URL: https://github.com/mattgroh/fitzpatrick17k
Size: 16,577 clinical images
Unique: Labeled by Fitzpatrick skin type (I-VI)
License: Research use (CC BY-NC-SA 4.0)

Categories Include:
├── Acne (various types)
├── Folliculitis
├── Post-inflammatory hyperpigmentation
├── Keloid
├── Contact dermatitis
├── Eczema
├── Psoriasis
└── Many more (114 conditions total)
```

**Download:**
```powershell
# Clone the repository
git clone https://github.com/mattgroh/fitzpatrick17k.git

# Download images (follow instructions in repo)
# Note: Requires accepting data use agreement
```

---

### 3. Acne Severity Dataset (Kaggle)
```
URL: https://www.kaggle.com/datasets/rutviklathiyateksworx/acne-severity-classification
Size: ~1,200 images
Labels: Mild, Moderate, Severe, Very Severe

Download:
kaggle datasets download -d rutviklathiyateksworx/acne-severity-classification
```

---

### 4. Skin Disease Dataset (Kaggle) - Good Variety
```
URL: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
Size: 4,500+ images
Categories: 10+ common conditions

Includes:
├── Acne
├── Eczema
├── Psoriasis
├── Tinea (ringworm)
├── Vitiligo
└── Others
```

---

### 5. SD-260 / SD-198 Dataset
```
Source: Academic (Sun et al.)
Size: 6,584 images, 198 categories
Access: Request from authors or search Kaggle mirrors

Paper: "Benchmark for Automatic Visual Classification of Clinical Skin Disease Images"
```

---

## 📥 DOWNLOAD SCRIPT

Save and run this script to download available datasets:

```python
"""
Dataset Downloader for Category 1 Skin Conditions
Run: python download_category1_datasets.py
"""

import os
import subprocess
import requests
from pathlib import Path

# Create directories
BASE_DIR = Path("c:/Users/Admin/Downloads/SKIN/data/category1")
DIRS = [
    "acne", "scars", "pigmentation", "dry_skin", 
    "bites", "wounds", "hair_issues", "other"
]

for d in DIRS:
    (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

print("✅ Directory structure created")

# Kaggle datasets to download
KAGGLE_DATASETS = [
    ("shubhamgoel27/dermnet", "dermnet"),
    ("rutviklathiyateksworx/acne-severity-classification", "acne_severity"),
    ("ismailpromus/skin-diseases-image-dataset", "skin_diseases"),
]

def download_kaggle_datasets():
    """Download datasets from Kaggle"""
    for dataset, folder in KAGGLE_DATASETS:
        dest = BASE_DIR / folder
        if not dest.exists():
            dest.mkdir(parents=True)
        
        print(f"📥 Downloading {dataset}...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", dataset, 
                "-p", str(dest),
                "--unzip"
            ], check=True)
            print(f"✅ Downloaded {dataset}")
        except Exception as e:
            print(f"❌ Failed to download {dataset}: {e}")
            print(f"   Manual download: https://www.kaggle.com/datasets/{dataset}")

if __name__ == "__main__":
    print("="*50)
    print("Category 1 Dataset Downloader")
    print("="*50)
    
    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        download_kaggle_datasets()
    except FileNotFoundError:
        print("⚠️ Kaggle CLI not found. Install with: pip install kaggle")
        print("   Then set up credentials: https://www.kaggle.com/docs/api")
        print("\n📋 Manual download links:")
        for dataset, _ in KAGGLE_DATASETS:
            print(f"   https://www.kaggle.com/datasets/{dataset}")
    
    print("\n✅ Done! Check:", BASE_DIR)
```

---

## 🗂️ MAPPING: Dataset → Category 1 Conditions

| Condition | Primary Dataset | Backup Dataset |
|-----------|-----------------|----------------|
| **Acne (all types)** | Acne Severity, DermNet | Fitzpatrick17k |
| **Whiteheads/Blackheads** | DermNet (Acne folder) | Custom collection |
| **Cystic Acne** | Acne Severity (Severe) | DermNet |
| **Acne Scars** | DermNet (Scars) | Custom collection |
| **Keloid Scars** | Fitzpatrick17k | DermNet |
| **PIH/PIE** | Fitzpatrick17k | Custom |
| **Age Spots** | DermNet | ISIC (pigmented lesions) |
| **Freckles** | Limited - Web scraping | Custom |
| **Milia** | DermNet | Custom |
| **Skin Tags** | DermNet | Custom |
| **Seborrheic Keratosis** | HAM10000, ISIC | DermNet |
| **Keratosis Pilaris** | DermNet | SD-198 |
| **Dry Skin/Eczema** | DermNet, Skin Diseases | Fitzpatrick17k |
| **Insect Bites** | Limited - DermNet | Custom collection |
| **Minor Burns** | Limited | Custom collection |
| **Bruises** | Very limited | Custom collection |

---

## ⚠️ CONDITIONS NEEDING CUSTOM DATA COLLECTION

These Category 1 conditions have **limited public datasets**:

```
NEEDS WEB SCRAPING OR CLINICAL PARTNERSHIP:
├── Insect bites (mosquito, bed bug, flea, spider)
├── Minor wounds (cuts, scrapes, bruises)
├── Sunburn (various degrees)
├── Friction blisters
├── Chapped lips
├── Cracked heels
├── Enlarged pores
├── Sebaceous filaments
├── Razor burn/bumps
└── Ingrown hairs
```

**Recommended approach:**
1. Partner with dermatology clinics
2. Use DermNet NZ web scraping (check terms)
3. Collect from medical image repositories
4. Use synthetic data augmentation

---

## 🔄 DATA ORGANIZATION STRUCTURE

After downloading, organize as:

```
data/
├── category1/
│   ├── acne/
│   │   ├── comedonal/
│   │   │   ├── whitehead/
│   │   │   └── blackhead/
│   │   ├── inflammatory/
│   │   │   ├── papule/
│   │   │   ├── pustule/
│   │   │   └── nodule/
│   │   ├── cystic/
│   │   ├── fungal/
│   │   └── hormonal/
│   ├── scars/
│   │   ├── icepick/
│   │   ├── rolling/
│   │   ├── boxcar/
│   │   ├── hypertrophic/
│   │   └── keloid/
│   ├── pigmentation/
│   │   ├── pih/
│   │   ├── pie/
│   │   ├── age_spots/
│   │   └── freckles/
│   ├── dry_skin/
│   │   ├── xerosis/
│   │   ├── keratosis_pilaris/
│   │   └── eczema_mild/
│   ├── oily_skin/
│   │   ├── enlarged_pores/
│   │   └── sebaceous_filaments/
│   ├── blemishes/
│   │   ├── milia/
│   │   ├── skin_tags/
│   │   ├── cherry_angioma/
│   │   └── seborrheic_keratosis/
│   ├── insect_bites/
│   ├── wounds/
│   └── hair_issues/
│       ├── ingrown_hairs/
│       ├── razor_burn/
│       └── alopecia/
```

---

## 📊 EXPECTED IMAGE COUNTS (After Collection)

| Subcategory | Target Images | Source Priority |
|-------------|---------------|-----------------|
| Acne (all) | 3,000+ | Kaggle + DermNet |
| Scars | 1,000+ | DermNet + Fitzpatrick |
| Pigmentation | 1,500+ | Fitzpatrick17k |
| Dry Skin | 800+ | DermNet |
| Blemishes | 500+ | DermNet + ISIC |
| Insect Bites | 500+ | Custom collection |
| Minor Wounds | 500+ | Custom collection |
| Hair Issues | 500+ | DermNet |

**Total Target: 8,000-10,000 images for Category 1**

---

## 🔗 DIRECT DOWNLOAD LINKS

### Kaggle (Requires Account)
1. DermNet: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
2. Acne Severity: https://www.kaggle.com/datasets/rutviklathiyateksworx/acne-severity-classification  
3. Skin Diseases: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
4. Acne Detection: https://www.kaggle.com/datasets/amitvkumar/acne-level-classification
5. Skin Condition: https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset

### GitHub
1. Fitzpatrick17k: https://github.com/mattgroh/fitzpatrick17k
2. ISIC Archive: https://www.isic-archive.com/

### Academic (Request Required)
1. ACNE04: Contact authors of paper
2. SD-198: Search for mirrors or contact authors

---

## ✅ NEXT STEPS

1. [ ] Set up Kaggle API credentials
2. [ ] Download DermNet dataset (largest, most useful)
3. [ ] Download Fitzpatrick17k (diversity)
4. [ ] Download Acne Severity dataset
5. [ ] Organize into folder structure above
6. [ ] Identify gaps in data coverage
7. [ ] Plan custom data collection for gaps
8. [ ] Create data preprocessing pipeline

---

## 📝 CITATION REQUIREMENTS

When using these datasets, cite:

```bibtex
@article{dermnet,
  title={DermNet NZ},
  url={https://dermnetnz.org/},
  note={New Zealand Dermatological Society}
}

@article{fitzpatrick17k,
  title={Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset},
  author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and others},
  journal={CVPR},
  year={2021}
}

@article{acne04,
  title={Joint Acne Image Grading and Counting via Label Distribution Learning},
  author={Wu, Xiaoping and others},
  journal={ICCV},
  year={2019}
}
```
