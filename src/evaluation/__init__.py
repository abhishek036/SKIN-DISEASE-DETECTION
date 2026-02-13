# Evaluation module
"""
Skin Disease Detection - Evaluation Package

Provides:
- ModelEvaluator: Comprehensive metrics and visualization
- GradCAM: Explainability via attention heatmaps
"""

from .metrics import (
    ModelEvaluator,
    evaluate_model,
)

from .gradcam import (
    GradCAM,
    visualize_gradcam,
    batch_gradcam,
    apply_colormap,
    overlay_heatmap,
)

__all__ = [
    'ModelEvaluator',
    'evaluate_model',
    'GradCAM',
    'visualize_gradcam',
    'batch_gradcam',
    'apply_colormap',
    'overlay_heatmap',
]