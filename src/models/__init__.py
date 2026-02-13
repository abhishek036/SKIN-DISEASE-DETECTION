# Models module
"""
Skin Disease Detection - Models Package

Available models:
- SkinLesionClassifier: EfficientNet-B0 based (recommended)
- EfficientNetB3Classifier: Higher accuracy, larger model
- MobileNetV3Classifier: For mobile deployment
"""

from .efficientnet import (
    SkinLesionClassifier,
    EfficientNetB3Classifier,
    MobileNetV3Classifier,
    build_model,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    'SkinLesionClassifier',
    'EfficientNetB3Classifier',
    'MobileNetV3Classifier',
    'build_model',
    'load_checkpoint',
    'save_checkpoint',
]