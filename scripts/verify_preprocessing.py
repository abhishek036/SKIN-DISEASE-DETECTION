"""
Verify preprocessing output.
"""
from pathlib import Path
import pandas as pd
from PIL import Image

processed_dir = Path('data/processed')

print("=" * 60)
print("PREPROCESSING VERIFICATION")
print("=" * 60)

for split in ['train', 'val', 'test']:
    split_dir = processed_dir / split
    labels_csv = split_dir / 'labels.csv'
    
    if not labels_csv.exists():
        print(f'ERROR: {labels_csv} not found')
        continue
    
    df = pd.read_csv(labels_csv)
    images = list(split_dir.glob('*.jpg'))
    
    print(f'\n{split.upper()}:')
    print(f'  Labels CSV entries: {len(df)}')
    print(f'  Images on disk: {len(images)}')
    print(f'  Match: {"YES" if len(df) == len(images) else "NO - MISMATCH!"}')
    
    # Check sample image
    if images:
        img = Image.open(images[0])
        print(f'  Image size: {img.size}')
        print(f'  Image mode: {img.mode}')
    
    # Class distribution
    print(f'  Class distribution:')
    for cls, count in df['class'].value_counts().items():
        pct = count / len(df) * 100
        print(f'    {cls}: {count} ({pct:.1f}%)')

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
