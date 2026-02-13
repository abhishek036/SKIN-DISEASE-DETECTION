"""
Skin Disease Detection - Optimized Training Script

Full training with best practices:
- Mixed precision (FP16) for faster training
- Gradient accumulation for larger effective batch size
- Cosine annealing with warm restarts
- Label smoothing
- Progressive unfreezing
- Better augmentation

Usage:
    python scripts/train_optimized.py
    python scripts/train_optimized.py --epochs 50 --batch-size 32
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.models import build_model, save_checkpoint
from src.data import (
    SkinLesionDataset,
    get_train_transforms,
    get_val_transforms,
)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for better generalization."""
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        
        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        
        # Apply smoothing
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Compute loss
        log_probs = F.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            log_probs = log_probs * self.weight.unsqueeze(0)
        
        loss = -(smooth_one_hot * log_probs).sum(dim=1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class OptimizedTrainer:
    """
    Optimized training pipeline with:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Label smoothing / Focal loss
    - Progressive learning rate
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor = None,
        device: str = 'cuda',
        log_dir: str = 'logs/optimized',
        checkpoint_dir: str = 'models/checkpoints/optimized',
        use_amp: bool = True,
        label_smoothing: float = 0.1,
        use_focal_loss: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        
        # Directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights.to(device) if class_weights is not None else None)
        else:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights.to(device) if class_weights is not None else None
            )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # State
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler = None,
        accumulation_steps: int = 1
    ) -> tuple:
        """Train one epoch with gradient accumulation."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            
            # Statistics
            running_loss += loss.item() * accumulation_steps * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / total, correct / total
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return running_loss / total, correct / total
    
    def train(
        self,
        num_epochs: int = 50,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        patience: int = 10,
        accumulation_steps: int = 2,
        warmup_epochs: int = 5
    ) -> dict:
        """Full training loop with OneCycleLR scheduler."""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # OneCycleLR for super-convergence
        total_steps = len(self.train_loader) * num_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_epochs / num_epochs,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []
        }
        
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("OPTIMIZED TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Accumulation Steps: {accumulation_steps}")
        print(f"Effective Batch Size: {self.train_loader.batch_size * accumulation_steps}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, scheduler, accumulation_steps)
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Log
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Check improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                patience_counter = 0
                
                save_checkpoint(
                    self.model, optimizer, epoch, val_loss,
                    {'val_acc': val_acc},
                    str(self.checkpoint_dir / 'best_model.pth')
                )
                print(f"  ✓ Best model saved (acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {elapsed/60:.1f} minutes")
        print(f"Best val accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}")
        
        self.writer.close()
        
        return history


def main():
    parser = argparse.ArgumentParser(description="Optimized training for skin lesion classifier")
    
    # Model
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'efficientnet_b3', 'mobilenet_v3'])
    parser.add_argument('--dropout', type=float, default=0.4)
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    
    # Regularization
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--focal-loss', action='store_true')
    parser.add_argument('--augmentation', type=str, default='strong',
                        choices=['light', 'medium', 'strong'])
    
    # Other
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    
    print(f"{'='*60}")
    print("SKIN DISEASE DETECTION - OPTIMIZED TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size} x {args.accumulation} = {args.batch_size * args.accumulation}")
    print(f"Epochs: {args.epochs}")
    print(f"Augmentation: {args.augmentation}")
    
    # Load data
    print("\nLoading data...")
    data_dir = Path(args.data_dir)
    
    train_labels = pd.read_csv(data_dir / 'train' / 'labels.csv')
    val_labels = pd.read_csv(data_dir / 'val' / 'labels.csv')
    
    train_transforms = get_train_transforms(
        image_size=args.image_size,
        augmentation_strength=args.augmentation
    )
    val_transforms = get_val_transforms(image_size=args.image_size)
    
    train_dataset = SkinLesionDataset(
        data_dir=data_dir / 'train',
        labels_df=train_labels,
        transform=train_transforms
    )
    
    val_dataset = SkinLesionDataset(
        data_dir=data_dir / 'val',
        labels_df=val_labels,
        transform=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    
    # Class weights
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights.numpy().round(2)}")
    
    # Build model
    print(f"\nBuilding model: {args.model}")
    model = build_model(
        model_name=args.model,
        num_classes=train_dataset.num_classes,
        pretrained=True,
        dropout_rate=args.dropout
    )
    print(f"Total params: {model.get_num_total_params():,}")
    
    # Train
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        use_amp=not args.no_amp,
        label_smoothing=args.label_smoothing,
        use_focal_loss=args.focal_loss
    )
    
    history = trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        accumulation_steps=args.accumulation,
        warmup_epochs=args.warmup
    )
    
    print(f"\nCheckpoints saved to: models/checkpoints/optimized/")
    print(f"TensorBoard logs: logs/optimized/")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir=logs/optimized")


if __name__ == "__main__":
    main()
