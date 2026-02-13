"""
Category 1 Dataset Downloader & Organizer
==========================================
Downloads and organizes datasets for everyday skin conditions.

Usage:
    python download_category1_datasets.py

Requirements:
    pip install kaggle requests tqdm Pillow
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict

# ============================================
# CONFIGURATION
# ============================================

# Use project-relative path
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "category1"
DOWNLOAD_DIR = BASE_DIR / "downloads"
ORGANIZED_DIR = BASE_DIR / "organized"

# Kaggle datasets for Category 1
KAGGLE_DATASETS = {
    "dermnet": {
        "id": "shubhamgoel27/dermnet",
        "description": "DermNet - 23 skin disease categories",
        "size": "~1.5 GB"
    },
    "acne_severity": {
        "id": "rutviklathiyateksworx/acne-severity-classification",
        "description": "Acne severity classification",
        "size": "~200 MB"
    },
    "skin_diseases": {
        "id": "ismailpromus/skin-diseases-image-dataset",
        "description": "10+ common skin conditions",
        "size": "~500 MB"
    },
    "acne_detection": {
        "id": "amitvkumar/acne-level-classification",
        "description": "Acne level classification",
        "size": "~100 MB"
    },
}

# Category 1 folder structure
CATEGORY1_STRUCTURE = {
    "acne": {
        "comedonal": ["whitehead", "blackhead"],
        "inflammatory": ["papule", "pustule", "nodule"],
        "cystic": [],
        "fungal": [],
        "hormonal": [],
        "back_acne": [],
    },
    "scars": {
        "icepick": [],
        "rolling": [],
        "boxcar": [],
        "atrophic": [],
        "hypertrophic": [],
        "keloid": [],
    },
    "pigmentation": {
        "pih": [],  # Post-inflammatory hyperpigmentation
        "pie": [],  # Post-inflammatory erythema
        "age_spots": [],
        "freckles": [],
        "melasma": [],
    },
    "dry_skin": {
        "xerosis": [],
        "keratosis_pilaris": [],
        "ichthyosis": [],
        "chapped_lips": [],
        "cracked_heels": [],
    },
    "oily_skin": {
        "seborrhea": [],
        "enlarged_pores": [],
        "sebaceous_filaments": [],
    },
    "blemishes": {
        "milia": [],
        "sebaceous_hyperplasia": [],
        "seborrheic_keratosis": [],
        "skin_tags": [],
        "cherry_angioma": [],
        "spider_veins": [],
    },
    "insect_bites": {
        "mosquito": [],
        "bed_bug": [],
        "flea": [],
        "spider": [],
        "bee_wasp": [],
        "tick": [],
        "scabies": [],
    },
    "wounds_trauma": {
        "abrasion": [],
        "laceration": [],
        "bruise": [],
        "blister": [],
        "sunburn": [],
        "minor_burn": [],
    },
    "hair_issues": {
        "ingrown_hair": [],
        "razor_burn": [],
        "razor_bumps": [],
        "folliculitis": [],
        "alopecia_areata": [],
    },
}

# Mapping from DermNet categories to our structure
DERMNET_MAPPING = {
    "Acne and Rosacea Photos": "acne/inflammatory",
    "Acne": "acne/inflammatory",
    "Atopic Dermatitis Photos": "dry_skin/xerosis",
    "Eczema Photos": "dry_skin/xerosis",
    "Nail Fungus and other Nail Disease": None,  # Category 8, not 1
    "Scars": "scars/hypertrophic",
    "Seborrheic Keratoses and other Benign Tumors": "blemishes/seborrheic_keratosis",
    "Psoriasis pictures Lichen Planus and related diseases": None,  # Category 3
    "Warts Molluscum and other Viral Infections": None,  # Category 2
    "Hair Loss Photos Alopecia and other Hair Diseases": "hair_issues/alopecia_areata",
    "Vascular Tumors": "blemishes/cherry_angioma",
    "Urticaria Hives": "insect_bites/mosquito",  # Similar appearance
}

# ============================================
# FUNCTIONS
# ============================================

def create_directory_structure():
    """Create the organized folder structure"""
    print("\n📁 Creating directory structure...")
    
    # Create base directories
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create category structure
    for category, subcategories in CATEGORY1_STRUCTURE.items():
        category_path = ORGANIZED_DIR / category
        category_path.mkdir(exist_ok=True)
        
        if isinstance(subcategories, dict):
            for subcat, sub_subcats in subcategories.items():
                subcat_path = category_path / subcat
                subcat_path.mkdir(exist_ok=True)
                
                for sub_sub in sub_subcats:
                    (subcat_path / sub_sub).mkdir(exist_ok=True)
    
    print("✅ Directory structure created")
    return True


def check_kaggle_setup() -> bool:
    """Check if Kaggle API is set up correctly"""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Kaggle CLI installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Kaggle CLI not found")
    print("\n📋 Setup instructions:")
    print("   1. pip install kaggle")
    print("   2. Go to kaggle.com → Account → Create API Token")
    print("   3. Save kaggle.json to C:\\Users\\<username>\\.kaggle\\")
    return False


def download_kaggle_dataset(dataset_key: str) -> bool:
    """Download a single Kaggle dataset"""
    if dataset_key not in KAGGLE_DATASETS:
        print(f"❌ Unknown dataset: {dataset_key}")
        return False
    
    dataset_info = KAGGLE_DATASETS[dataset_key]
    dataset_id = dataset_info["id"]
    dest_dir = DOWNLOAD_DIR / dataset_key
    
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"⏭️ {dataset_key} already downloaded, skipping...")
        return True
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {dataset_key}...")
    print(f"   ID: {dataset_id}")
    print(f"   Size: {dataset_info['size']}")
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest_dir), "--unzip"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Downloaded {dataset_key}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_all_datasets():
    """Download all configured Kaggle datasets"""
    print("\n" + "="*50)
    print("DOWNLOADING KAGGLE DATASETS")
    print("="*50)
    
    if not check_kaggle_setup():
        return False
    
    success_count = 0
    for dataset_key in KAGGLE_DATASETS:
        if download_kaggle_dataset(dataset_key):
            success_count += 1
    
    print(f"\n📊 Downloaded {success_count}/{len(KAGGLE_DATASETS)} datasets")
    return success_count > 0


def count_images_in_dir(path: Path) -> int:
    """Count image files in a directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    count = 0
    if path.exists():
        for f in path.rglob('*'):
            if f.suffix.lower() in extensions:
                count += 1
    return count


def generate_dataset_report():
    """Generate a report of downloaded datasets"""
    print("\n" + "="*50)
    print("DATASET REPORT")
    print("="*50)
    
    total_images = 0
    
    # Check downloads
    print("\n📥 Downloaded Datasets:")
    for dataset_key in KAGGLE_DATASETS:
        path = DOWNLOAD_DIR / dataset_key
        count = count_images_in_dir(path)
        total_images += count
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {dataset_key}: {count:,} images")
    
    # Check organized structure
    print("\n📁 Organized Structure:")
    for category in CATEGORY1_STRUCTURE:
        path = ORGANIZED_DIR / category
        count = count_images_in_dir(path)
        print(f"   {category}: {count:,} images")
    
    print(f"\n📊 Total images available: {total_images:,}")
    
    return total_images


def print_manual_download_links():
    """Print links for manual download"""
    print("\n" + "="*50)
    print("MANUAL DOWNLOAD LINKS")
    print("="*50)
    
    print("\n🔗 Kaggle Datasets (requires account):")
    for key, info in KAGGLE_DATASETS.items():
        print(f"\n   {key}:")
        print(f"   URL: https://www.kaggle.com/datasets/{info['id']}")
        print(f"   {info['description']} ({info['size']})")
    
    print("\n🔗 Other Sources:")
    print("\n   Fitzpatrick17k (ESSENTIAL for diversity):")
    print("   URL: https://github.com/mattgroh/fitzpatrick17k")
    print("   Must accept data use agreement")
    
    print("\n   DermNet NZ (Reference images):")
    print("   URL: https://dermnetnz.org/image-library")
    print("   Check terms before scraping")
    
    print("\n   ISIC Archive (Skin cancer focus):")
    print("   URL: https://www.isic-archive.com/")


def main():
    """Main function"""
    print("="*60)
    print("🔬 CATEGORY 1 DATASET DOWNLOADER")
    print("   Everyday Skin Issues - Data Acquisition")
    print("="*60)
    
    # Create directory structure
    create_directory_structure()
    
    # Check arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--links":
            print_manual_download_links()
            return
        elif sys.argv[1] == "--report":
            generate_dataset_report()
            return
        elif sys.argv[1] == "--download":
            download_all_datasets()
            generate_dataset_report()
            return
    
    # Default: show menu
    print("\n📋 Options:")
    print("   1. Download all Kaggle datasets")
    print("   2. Show manual download links")
    print("   3. Generate dataset report")
    print("   4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        download_all_datasets()
        generate_dataset_report()
    elif choice == "2":
        print_manual_download_links()
    elif choice == "3":
        generate_dataset_report()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    main()
