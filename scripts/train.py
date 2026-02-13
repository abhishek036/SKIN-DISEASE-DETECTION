"""
Skin Disease Detection - Training Script

Usage:
    python scripts/train.py
    python scripts/train.py --model efficientnet_b3 --epochs 50
    python scripts/train.py --two-phase --phase1-epochs 10 --phase2-epochs 40
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models import build_model
from src.data import create_dataloaders, get_class_info
from src.training.trainer import Trainer, train_two_phase


def main():
    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
    
    # Model
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'efficientnet_b3', 'mobilenet_v3'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loader workers')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    # Two-phase training
    parser.add_argument('--two-phase', action='store_true',
                        help='Use two-phase training (recommended)')
    parser.add_argument('--phase1-epochs', type=int, default=10,
                        help='Phase 1 epochs (head only)')
    parser.add_argument('--phase2-epochs', type=int, default=40,
                        help='Phase 2 epochs (fine-tuning)')
    parser.add_argument('--phase1-lr', type=float, default=1e-3,
                        help='Phase 1 learning rate')
    parser.add_argument('--phase2-lr', type=float, default=1e-4,
                        help='Phase 2 learning rate')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Set seed
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 60)
    print("SKIN DISEASE DETECTION - TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}")
    print(f"  Two-phase: {args.two_phase}")
    
    # Check for GPU
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Create data loaders
    print("\nLoading data...")
    data_dir = Path(args.data_dir)
    
    dataloaders = create_dataloaders(
        train_dir=data_dir / 'train',
        val_dir=data_dir / 'val',
        test_dir=data_dir / 'test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_weighted_sampling=True
    )
    
    # Get class info
    class_info = get_class_info(data_dir / 'train')
    num_classes = class_info['num_classes']
    
    print(f"  Classes: {num_classes}")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    
    # Compute class weights
    class_weights = torch.FloatTensor([
        class_info['class_weights'].get(cls, 1.0)
        for cls in sorted(class_info['class_weights'].keys())
    ])
    print(f"  Class weights: {class_weights.tolist()}")
    
    # Build model
    print(f"\nBuilding model: {args.model}")
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout
    )
    
    print(f"  Total params: {model.get_num_total_params():,}")
    print(f"  Trainable params: {model.get_num_trainable_params():,}")
    
    # Train
    if args.two_phase:
        history = train_two_phase(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            class_weights=class_weights,
            device=device,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase1_lr=args.phase1_lr,
            phase2_lr=args.phase2_lr,
            patience=args.patience
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            class_weights=class_weights,
            device=device
        )
        
        history = trainer.train(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nCheckpoints saved to: models/checkpoints/")
    print("TensorBoard logs saved to: logs/")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir=logs")


if __name__ == "__main__":
    main()
