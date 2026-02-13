"""
Skin Disease Detection - Evaluation Module

Comprehensive model evaluation with:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix visualization
- Per-class performance analysis
- Grad-CAM explainability
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation for skin lesion classification.
    
    Generates:
        - Classification metrics
        - Confusion matrix
        - ROC curves
        - Per-class analysis
        - Grad-CAM visualizations
    """
    
    CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    
    CLASS_FULL_NAMES = {
        'AK': 'Actinic Keratosis',
        'BCC': 'Basal Cell Carcinoma',
        'BKL': 'Benign Keratosis',
        'DF': 'Dermatofibroma',
        'MEL': 'Melanoma',
        'NV': 'Melanocytic Nevus',
        'SCC': 'Squamous Cell Carcinoma',
        'VASC': 'Vascular Lesion',
    }
    
    # Clinical severity for prioritization
    CLASS_SEVERITY = {
        'MEL': 'critical',   # Melanoma - most dangerous
        'SCC': 'high',       # Squamous Cell Carcinoma
        'BCC': 'high',       # Basal Cell Carcinoma
        'AK': 'medium',      # Actinic Keratosis (precancer)
        'BKL': 'low',        # Benign Keratosis
        'NV': 'low',         # Melanocytic Nevus
        'DF': 'low',         # Dermatofibroma
        'VASC': 'low',       # Vascular Lesion
    }
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cuda',
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names or self.CLASS_NAMES
        self.num_classes = len(self.class_names)
        
        # Results storage
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
    @torch.no_grad()
    def run_inference(self) -> None:
        """Run inference on entire dataset."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
        for images, labels in tqdm(self.dataloader, desc="Evaluating"):
            images = images.to(self.device)
            
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(labels.numpy())
            self.all_probs.extend(probs.cpu().numpy())
        
        self.all_preds = np.array(self.all_preds)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)
        
    def compute_metrics(self) -> Dict:
        """Compute comprehensive classification metrics."""
        if len(self.all_preds) == 0:
            self.run_inference()
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(self.all_labels, self.all_preds)
        metrics['macro_precision'] = precision_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        )
        metrics['macro_recall'] = recall_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        )
        metrics['macro_f1'] = f1_score(
            self.all_labels, self.all_preds, average='macro', zero_division=0
        )
        metrics['weighted_f1'] = f1_score(
            self.all_labels, self.all_preds, average='weighted', zero_division=0
        )
        
        # AUC (one-vs-rest)
        try:
            metrics['macro_auc'] = roc_auc_score(
                self.all_labels, self.all_probs, multi_class='ovr', average='macro'
            )
            metrics['weighted_auc'] = roc_auc_score(
                self.all_labels, self.all_probs, multi_class='ovr', average='weighted'
            )
        except ValueError:
            metrics['macro_auc'] = 0.0
            metrics['weighted_auc'] = 0.0
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i, cls in enumerate(self.class_names):
            cls_mask = self.all_labels == i
            cls_preds = self.all_preds[cls_mask]
            cls_labels = self.all_labels[cls_mask]
            
            if len(cls_labels) > 0:
                cls_correct = (cls_preds == cls_labels).sum()
                cls_recall = cls_correct / len(cls_labels)
            else:
                cls_recall = 0.0
            
            # Precision: of all predicted as this class, how many are correct
            pred_mask = self.all_preds == i
            if pred_mask.sum() > 0:
                cls_precision = (self.all_labels[pred_mask] == i).sum() / pred_mask.sum()
            else:
                cls_precision = 0.0
            
            # F1
            if cls_precision + cls_recall > 0:
                cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall)
            else:
                cls_f1 = 0.0
            
            metrics['per_class'][cls] = {
                'precision': float(cls_precision),
                'recall': float(cls_recall),
                'f1': float(cls_f1),
                'support': int(cls_mask.sum()),
                'severity': self.CLASS_SEVERITY.get(cls, 'unknown')
            }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if len(self.all_preds) == 0:
            self.run_inference()
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            save_path: Path to save figure
            normalize: Normalize by row (true labels)
            figsize: Figure size
        """
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curves(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Plot ROC curves for each class.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        if len(self.all_probs) == 0:
            self.run_inference()
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        
        for i, (cls, color) in enumerate(zip(self.class_names, colors)):
            # Binary labels for this class
            y_true = (self.all_labels == i).astype(int)
            y_score = self.all_probs[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        plt.close()
    
    def print_classification_report(self) -> str:
        """Print sklearn classification report."""
        if len(self.all_preds) == 0:
            self.run_inference()
        
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            digits=4
        )
        print(report)
        return report
    
    def print_summary(self) -> None:
        """Print evaluation summary."""
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
        print(f"  Macro AUC:        {metrics['macro_auc']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print("-" * 60)
        print(f"{'Class':<8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8} {'Severity':>10}")
        print("-" * 60)
        
        for cls in self.class_names:
            pc = metrics['per_class'][cls]
            print(f"{cls:<8} {pc['precision']:>8.4f} {pc['recall']:>8.4f} "
                  f"{pc['f1']:>8.4f} {pc['support']:>8} {pc['severity']:>10}")
        
        # Highlight critical class performance
        print("\n" + "-" * 60)
        print("CRITICAL CLASS PERFORMANCE (Cancer Detection):")
        for cls in ['MEL', 'BCC', 'SCC']:
            pc = metrics['per_class'].get(cls, {})
            print(f"  {cls} ({self.CLASS_FULL_NAMES.get(cls, cls)}): "
                  f"Recall = {pc.get('recall', 0):.4f}")
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all evaluation results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference if not done
        if len(self.all_preds) == 0:
            self.run_inference()
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Save metrics JSON
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save confusion matrix
        self.plot_confusion_matrix(
            save_path=str(output_dir / 'confusion_matrix.png'),
            normalize=True
        )
        self.plot_confusion_matrix(
            save_path=str(output_dir / 'confusion_matrix_counts.png'),
            normalize=False
        )
        
        # Save ROC curves
        self.plot_roc_curves(save_path=str(output_dir / 'roc_curves.png'))
        
        # Save classification report
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            digits=4
        )
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to {output_dir}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    output_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained model
        dataloader: Test/validation dataloader
        device: Device to use
        output_dir: Directory to save results (optional)
        class_names: List of class names
        
    Returns:
        Metrics dictionary
    """
    evaluator = ModelEvaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names
    )
    
    evaluator.run_inference()
    evaluator.print_summary()
    evaluator.print_classification_report()
    
    if output_dir:
        evaluator.save_results(output_dir)
    
    return evaluator.compute_metrics()
