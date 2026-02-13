"""
Complete Dataset Setup Script
==============================
Downloads and organizes ALL datasets for skin disease detection system.

Covers:
- Category 1: Everyday skin issues (acne, scars, etc.)
- Category 2-4: Medical conditions (eczema, infections, serious diseases)
- Ensures diversity across skin tones (Fitzpatrick I-VI)

Usage:
    python setup_datasets.py --download-all
    python setup_datasets.py --organize
    python setup_datasets.py --verify

Requirements:
    pip install kaggle requests tqdm Pillow pandas
"""

import os
import sys
import json
import subprocess
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict
import argparse
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️ Install tqdm for progress bars: pip install tqdm")

# ============================================
# CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"

# Create directories
for path in [RAW_DATA, PROCESSED_DATA]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================
# DATASET REGISTRY
# ============================================

DATASETS = {
    # === CATEGORY 1: EVERYDAY ISSUES ===
    "dermnet": {
        "source": "kaggle",
        "id": "shubhamgoel27/dermnet",
        "size_gb": 1.5,
        "images": 19500,
        "categories": 23,
        "priority": "CRITICAL",
        "covers": ["acne", "eczema", "psoriasis", "scars", "keratosis"],
        "description": "Comprehensive dermatology image database"
    },
    
    "acne_severity": {
        "source": "kaggle",
        "id": "rutviklathiyateksworx/acne-severity-classification",
        "size_gb": 0.2,
        "images": 1200,
        "categories": 4,
        "priority": "HIGH",
        "covers": ["acne_mild", "acne_moderate", "acne_severe"],
        "description": "Acne severity grading dataset"
    },
    
    "skin_diseases": {
        "source": "kaggle",
        "id": "ismailpromus/skin-diseases-image-dataset",
        "size_gb": 0.5,
        "images": 4500,
        "categories": 10,
        "priority": "HIGH",
        "covers": ["common_conditions", "infections"],
        "description": "10+ common skin conditions"
    },
    
    # === DIVERSITY & REPRESENTATION ===
    "fitzpatrick17k": {
        "source": "github",
        "url": "https://github.com/mattgroh/fitzpatrick17k.git",
        "size_gb": 2.0,
        "images": 16577,
        "categories": 114,
        "priority": "CRITICAL",
        "covers": ["all_skin_tones", "fitzpatrick_labeled"],
        "description": "Diverse dataset with Fitzpatrick skin type labels"
    },
    
    # === ALREADY DOWNLOADED ===
    "ham10000": {
        "source": "local",
        "path": "skin-cancer-mnist-ham10000",
        "images": 10015,
        "categories": 7,
        "priority": "EXISTING",
        "covers": ["melanoma", "bcc", "akiec", "bkl", "df", "nv", "vasc"],
        "description": "Skin cancer dataset (already downloaded)"
    },
    
    "isic2019": {
        "source": "local",
        "path": "ISIC_2019_Training_Input",
        "images": 25331,
        "categories": 8,
        "priority": "EXISTING",
        "covers": ["melanoma", "nv", "bcc", "akiec", "bkl", "df", "vasc", "scc"],
        "description": "ISIC 2019 challenge dataset (already downloaded)"
    },
    
    # === OPTIONAL/SUPPLEMENTARY ===
    "acne_detection": {
        "source": "kaggle",
        "id": "amitvkumar/acne-level-classification",
        "size_gb": 0.1,
        "images": 800,
        "categories": 4,
        "priority": "MEDIUM",
        "covers": ["acne_classification"],
        "description": "Additional acne detection dataset"
    },
    
    "monkeypox": {
        "source": "kaggle",
        "id": "nafin59/monkeypox-skin-lesion-dataset",
        "size_gb": 0.3,
        "images": 2000,
        "categories": 2,
        "priority": "LOW",
        "covers": ["monkeypox", "chickenpox", "measles"],
        "description": "Viral skin infections"
    },
}

# ============================================
# DOWNLOAD FUNCTIONS
# ============================================

def check_kaggle_setup():
    """Verify Kaggle CLI is configured"""
    try:
        result = subprocess.run(
            ["kaggle", "--version"], 
            capture_output=True, 
            text=True,
            check=True
        )
        print("✅ Kaggle CLI found:", result.stdout.strip())
        return True
    except FileNotFoundError:
        print("❌ Kaggle CLI not found")
        print("   Install: pip install kaggle")
        print("   Setup: https://www.kaggle.com/docs/api")
        return False
    except subprocess.CalledProcessError:
        print("❌ Kaggle CLI error - check credentials")
        return False

def download_kaggle_dataset(dataset_id: str, dest: Path, unzip: bool = True):
    """Download dataset from Kaggle"""
    print(f"\n📥 Downloading: {dataset_id}")
    
    dest.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_id,
        "-p", str(dest),
    ]
    
    if unzip:
        cmd.append("--unzip")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Downloaded: {dataset_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"   Manual download: https://www.kaggle.com/datasets/{dataset_id}")
        return False

def clone_github_dataset(url: str, dest: Path):
    """Clone GitHub repository"""
    print(f"\n📥 Cloning: {url}")
    
    try:
        subprocess.run(
            ["git", "clone", url, str(dest)],
            check=True
        )
        print(f"✅ Cloned: {url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False

def download_all_datasets(priority_filter: str = None):
    """Download all datasets matching priority"""
    print("\n" + "="*60)
    print("DATASET DOWNLOAD MANAGER")
    print("="*60)
    
    # Check prerequisites
    if not check_kaggle_setup():
        print("\n⚠️ Cannot download Kaggle datasets without setup")
        return
    
    # Filter datasets
    to_download = {
        name: info for name, info in DATASETS.items()
        if info.get("source") != "local" and 
        (priority_filter is None or info.get("priority") == priority_filter)
    }
    
    print(f"\n📋 Datasets to download: {len(to_download)}")
    
    total_size = sum(d.get("size_gb", 0) for d in to_download.values())
    print(f"💾 Total size: ~{total_size:.1f} GB")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Download each dataset
    success_count = 0
    for name, info in to_download.items():
        dest = RAW_DATA / name
        
        if dest.exists():
            print(f"\n⏭️ Skipping {name} (already exists)")
            continue
        
        if info["source"] == "kaggle":
            if download_kaggle_dataset(info["id"], dest):
                success_count += 1
        
        elif info["source"] == "github":
            if clone_github_dataset(info["url"], dest):
                success_count += 1
    
    print(f"\n✅ Downloaded: {success_count}/{len(to_download)} datasets")

# ============================================
# ORGANIZATION FUNCTIONS
# ============================================

def organize_datasets():
    """Organize downloaded datasets into unified structure"""
    print("\n" + "="*60)
    print("DATASET ORGANIZATION")
    print("="*60)
    
    # Define target structure
    target_structure = {
        "category1_everyday": [
            "acne_comedonal", "acne_inflammatory", "acne_cystic",
            "scars_atrophic", "scars_hypertrophic", "scars_keloid",
            "pigmentation_pih", "pigmentation_melasma", "pigmentation_freckles",
            "dry_skin_eczema", "dry_skin_keratosis_pilaris",
            "blemishes_skin_tags", "blemishes_milia", "blemishes_cherry_angioma"
        ],
        "category2_medical": [
            "eczema_atopic", "psoriasis", "rosacea", "hives",
            "fungal_ringworm", "fungal_candida", "bacterial_impetigo",
            "vitiligo", "alopecia_areata"
        ],
        "category3_serious": [
            "melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma",
            "actinic_keratosis", "merkel_cell", "kaposi_sarcoma"
        ],
        "category4_rare": [
            "pemphigus", "lupus_cutaneous", "scleroderma",
            "dermatomyositis", "vasculitis"
        ]
    }
    
    # Create target directories
    for category, conditions in target_structure.items():
        category_path = PROCESSED_DATA / category
        for condition in conditions:
            (category_path / condition).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created unified folder structure")
    
    # TODO: Implement actual file organization logic
    # This would map images from raw datasets to organized structure
    print("\n⚠️ Manual organization required - see mapping guide")

# ============================================
# VERIFICATION FUNCTIONS
# ============================================

def verify_downloads():
    """Verify all expected datasets are present"""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    summary = {
        "total_datasets": len(DATASETS),
        "downloaded": 0,
        "missing": 0,
        "total_images": 0
    }
    
    print("\n📊 Dataset Status:\n")
    
    for name, info in DATASETS.items():
        if info.get("source") == "local":
            path = PROJECT_ROOT / info["path"]
        else:
            path = RAW_DATA / name
        
        exists = path.exists()
        status = "✅" if exists else "❌"
        
        print(f"{status} {name:20} | {info['priority']:8} | {info['images']:6} images | {info.get('size_gb', 0):4.1f} GB")
        
        if exists:
            summary["downloaded"] += 1
            summary["total_images"] += info["images"]
        else:
            summary["missing"] += 1
    
    print("\n" + "-"*60)
    print(f"Downloaded: {summary['downloaded']}/{summary['total_datasets']}")
    print(f"Total images: ~{summary['total_images']:,}")
    print(f"Missing: {summary['missing']}")
    
    if summary["missing"] > 0:
        print("\n💡 Run with --download-all to get missing datasets")

def generate_dataset_report():
    """Generate comprehensive dataset report"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "datasets": DATASETS,
        "statistics": {
            "total_datasets": len(DATASETS),
            "total_images": sum(d["images"] for d in DATASETS.values()),
            "total_size_gb": sum(d.get("size_gb", 0) for d in DATASETS.values()),
            "conditions_covered": []
        }
    }
    
    # Collect all unique conditions
    all_conditions = set()
    for info in DATASETS.values():
        all_conditions.update(info.get("covers", []))
    report["statistics"]["conditions_covered"] = sorted(all_conditions)
    
    # Save report
    report_path = DATA_ROOT / "dataset_inventory.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {report_path}")

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup datasets for skin disease detection"
    )
    
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all missing datasets"
    )
    
    parser.add_argument(
        "--download-critical",
        action="store_true",
        help="Download only CRITICAL priority datasets"
    )
    
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize datasets into unified structure"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset downloads"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate dataset inventory report"
    )
    
    args = parser.parse_args()
    
    # Default action if no args
    if not any(vars(args).values()):
        verify_downloads()
        return
    
    # Execute requested actions
    if args.download_critical:
        download_all_datasets(priority_filter="CRITICAL")
    
    if args.download_all:
        download_all_datasets()
    
    if args.organize:
        organize_datasets()
    
    if args.verify:
        verify_downloads()
    
    if args.report:
        generate_dataset_report()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelled by user")
        sys.exit(0)
