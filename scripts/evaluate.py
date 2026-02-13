"""
Skin Disease Detection - Evaluation Script

Evaluate trained model on test set with comprehensive metrics.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/phase2/best_model.pth
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth --gradcam
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from src.models import build_model, load_checkpoint
from src.data import SkinLesionDataset, get_val_transforms
from src.evaluation import ModelEvaluator, visualize_gradcam
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate skin lesion classifier")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed/test",
        help="Test data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model", type=str, default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b3", "mobilenet_v3"],
        help="Model architecture"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Image size"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generate Grad-CAM visualizations for sample images"
    )
    parser.add_argument(
        "--num-gradcam", type=int, default=10,
        help="Number of Grad-CAM samples per class"
    )
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("SKIN DISEASE DETECTION - MODEL EVALUATION")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.data_dir}")
    
    # Load test data
    test_dir = Path(args.data_dir)
    test_labels = pd.read_csv(test_dir / "labels.csv")
    
    transform = get_val_transforms(image_size=args.image_size)
    
    test_dataset = SkinLesionDataset(
        data_dir=test_dir,
        labels_df=test_labels,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {test_dataset.num_classes}")
    
    # Build model
    model = build_model(
        model_name=args.model,
        num_classes=test_dataset.num_classes,
        pretrained=False
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = load_checkpoint(model, args.checkpoint, device=device)
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('loss', 'N/A'):.4f}" if 'loss' in checkpoint else "")
    
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("\nRunning evaluation...")
    evaluator = ModelEvaluator(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=test_dataset.DEFAULT_CLASSES
    )
    
    evaluator.run_inference()
    evaluator.print_summary()
    
    print("\nClassification Report:")
    print("-" * 60)
    evaluator.print_classification_report()
    
    # Save results
    output_dir = Path(args.output_dir)
    evaluator.save_results(str(output_dir))
    
    # Grad-CAM visualizations
    if args.gradcam:
        print("\nGenerating Grad-CAM visualizations...")
        gradcam_dir = output_dir / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sample images from each class
        for cls_idx, cls_name in enumerate(test_dataset.DEFAULT_CLASSES):
            cls_mask = test_labels["class"] == cls_name
            cls_images = test_labels[cls_mask]["image"].tolist()[:args.num_gradcam]
            
            for img_id in cls_images:
                img_path = test_dir / f"{img_id}.jpg"
                if img_path.exists():
                    save_path = gradcam_dir / f"{cls_name}_{img_id}_gradcam.png"
                    visualize_gradcam(
                        model=model,
                        image_path=str(img_path),
                        transform=transform,
                        class_names=test_dataset.DEFAULT_CLASSES,
                        device=device,
                        save_path=str(save_path)
                    )
        
        print(f"Grad-CAM visualizations saved to {gradcam_dir}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
