"""
Skin Disease Detection - Training Module

Complete training pipeline with:
- Class-weighted loss for imbalanced data
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- TensorBoard logging
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import build_model, save_checkpoint


class Trainer:
    """
    Training orchestrator for skin lesion classification.
    
    Handles:
        - Two-phase training (frozen backbone → fine-tuning)
        - Class-weighted cross-entropy loss
        - Learning rate scheduling
        - Early stopping
        - Checkpoint management
        - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        log_dir: str = 'logs',
        checkpoint_dir: str = 'models/checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(
        self,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 7,
        min_lr: float = 1e-6,
        scheduler_type: str = 'plateau'
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Maximum epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            min_lr: Minimum learning rate
            scheduler_type: 'plateau' or 'cosine'
            
        Returns:
            Training history dict
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5,
                patience=3, min_lr=min_lr
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=min_lr
            )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset):,}")
        print(f"Val samples: {len(self.val_loader.dataset):,}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss, 'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc, 'val': val_acc
            }, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                # Save best model
                save_checkpoint(
                    self.model, optimizer, epoch, val_loss,
                    {'val_acc': val_acc, 'train_acc': train_acc},
                    str(self.checkpoint_dir / 'best_model.pth')
                )
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"Training complete in {elapsed/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val acc: {self.best_val_acc:.4f}")
        
        # Save final model
        save_checkpoint(
            self.model, optimizer, self.current_epoch, val_loss,
            {'val_acc': val_acc},
            str(self.checkpoint_dir / 'final_model.pth')
        )
        
        self.writer.close()
        
        return history


def train_two_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    phase1_epochs: int = 10,
    phase2_epochs: int = 40,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    patience: int = 7
) -> Dict:
    """
    Two-phase training strategy.
    
    Phase 1: Train classification head only (backbone frozen)
    Phase 2: Fine-tune entire model with lower learning rate
    
    Args:
        model: Model with freeze_backbone/unfreeze_backbone methods
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Optional class weights for loss
        device: Training device
        phase1_epochs: Epochs for head training
        phase2_epochs: Epochs for fine-tuning
        phase1_lr: Learning rate for phase 1
        phase2_lr: Learning rate for phase 2
        patience: Early stopping patience
        
    Returns:
        Combined training history
    """
    print("=" * 60)
    print("TWO-PHASE TRAINING")
    print("=" * 60)
    
    # Phase 1: Train head only
    print("\n" + "=" * 60)
    print("PHASE 1: Training Classification Head")
    print("=" * 60)
    
    model.freeze_backbone()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        log_dir='logs/phase1',
        checkpoint_dir='models/checkpoints/phase1'
    )
    
    history1 = trainer.train(
        num_epochs=phase1_epochs,
        learning_rate=phase1_lr,
        patience=patience
    )
    
    # Phase 2: Fine-tune
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning Entire Model")
    print("=" * 60)
    
    model.unfreeze_backbone(num_layers=20)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device,
        log_dir='logs/phase2',
        checkpoint_dir='models/checkpoints/phase2'
    )
    
    history2 = trainer.train(
        num_epochs=phase2_epochs,
        learning_rate=phase2_lr,
        patience=patience
    )
    
    # Combine histories
    combined_history = {
        'phase1': history1,
        'phase2': history2
    }
    
    return combined_history
