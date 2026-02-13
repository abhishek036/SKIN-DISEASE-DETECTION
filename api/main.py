"""
Skin Disease Detection - FastAPI Inference Server

REST API for skin lesion classification with:
- Image upload and prediction
- Confidence scores
- Top-K predictions
- Grad-CAM explainability
- Health checks

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import base64
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, load_checkpoint
from src.data import get_val_transforms
from src.evaluation.gradcam import GradCAM, apply_colormap, overlay_heatmap


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "efficientnet_b0"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "phase2" / "best_model.pth"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

CLASS_INFO = {
    'AK': {
        'name': 'Actinic Keratosis',
        'category': 'precancer',
        'severity': 'medium',
        'description': 'Pre-cancerous scaly patches caused by sun damage',
        'recommendation': 'Consult dermatologist within 1 month'
    },
    'BCC': {
        'name': 'Basal Cell Carcinoma',
        'category': 'cancer',
        'severity': 'high',
        'description': 'Most common type of skin cancer, rarely spreads',
        'recommendation': 'Consult dermatologist within 2 weeks'
    },
    'BKL': {
        'name': 'Benign Keratosis',
        'category': 'benign',
        'severity': 'low',
        'description': 'Non-cancerous skin growth, often age-related',
        'recommendation': 'Routine monitoring, no urgent action needed'
    },
    'DF': {
        'name': 'Dermatofibroma',
        'category': 'benign',
        'severity': 'low',
        'description': 'Harmless fibrous nodule in the skin',
        'recommendation': 'No treatment needed unless symptomatic'
    },
    'MEL': {
        'name': 'Melanoma',
        'category': 'cancer',
        'severity': 'critical',
        'description': 'Most dangerous form of skin cancer',
        'recommendation': 'URGENT: Consult dermatologist immediately'
    },
    'NV': {
        'name': 'Melanocytic Nevus',
        'category': 'benign',
        'severity': 'low',
        'description': 'Common mole, usually harmless',
        'recommendation': 'Monitor for changes (ABCDE rule)'
    },
    'SCC': {
        'name': 'Squamous Cell Carcinoma',
        'category': 'cancer',
        'severity': 'high',
        'description': 'Second most common skin cancer',
        'recommendation': 'Consult dermatologist within 1 week'
    },
    'VASC': {
        'name': 'Vascular Lesion',
        'category': 'benign',
        'severity': 'low',
        'description': 'Blood vessel abnormality in the skin',
        'recommendation': 'Routine evaluation if symptomatic'
    }
}


# =============================================================================
# Response Models
# =============================================================================

class PredictionResult(BaseModel):
    class_code: str
    class_name: str
    confidence: float
    category: str
    severity: str
    description: str
    recommendation: str


class PredictionResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    top_k_predictions: List[Dict]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Skin Disease Detection API",
    description="AI-powered skin lesion classification for screening purposes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend
STATIC_DIR = PROJECT_ROOT / "api" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Global Model State
# =============================================================================

model = None
transform = None
gradcam = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, transform, gradcam
    
    print(f"Loading model on {DEVICE}...")
    
    model = build_model(
        model_name=MODEL_NAME,
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )
    
    if CHECKPOINT_PATH.exists():
        load_checkpoint(model, str(CHECKPOINT_PATH), device=DEVICE)
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Using random weights - predictions will be meaningless!")
    
    model = model.to(DEVICE)
    model.eval()
    
    transform = get_val_transforms(image_size=IMAGE_SIZE)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    print("Model loaded successfully!")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Skin Disease Detection API</h1><p>Visit /docs for API documentation</p>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=DEVICE,
        model_name=MODEL_NAME
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=DEVICE,
        model_name=MODEL_NAME
    )


@app.get("/classes")
async def get_classes():
    """Get information about all classes."""
    return {
        "classes": CLASS_INFO,
        "total": len(CLASS_NAMES)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = 3
):
    """
    Predict skin lesion class from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        top_k: Number of top predictions to return
        
    Returns:
        Prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Get prediction
        pred_idx = probs.argmax().item()
        pred_code = CLASS_NAMES[pred_idx]
        pred_info = CLASS_INFO[pred_code]
        
        # All probabilities for the frontend
        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        
        # Simple response for frontend
        return {
            "predicted_class": pred_code,
            "confidence": float(probs[pred_idx]),
            "class_name": pred_info['name'],
            "category": pred_info['category'],
            "severity": pred_info['severity'],
            "description": pred_info['description'],
            "recommendation": pred_info['recommendation'],
            "all_probabilities": all_probs,
            "disclaimer": "This is an AI screening tool for educational purposes only. "
                         "It is NOT a medical diagnosis. Always consult a qualified "
                         "healthcare professional for proper evaluation and treatment."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_with_gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...)
):
    """
    Predict with Grad-CAM visualization.
    
    Returns prediction plus base64-encoded heatmap overlay.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_array = np.array(image)
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get prediction and Grad-CAM
        heatmap, pred_class, confidence = gradcam(input_tensor)
        
        # Create overlay
        import cv2
        resized_original = cv2.resize(original_array, (heatmap.shape[1], heatmap.shape[0]))
        overlay = overlay_heatmap(resized_original, heatmap, alpha=0.5)
        
        # Encode overlay as base64
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        overlay_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        pred_code = CLASS_NAMES[pred_class]
        pred_info = CLASS_INFO[pred_code]
        
        return {
            "success": True,
            "prediction": {
                "class_code": pred_code,
                "class_name": pred_info['name'],
                "confidence": float(confidence),
                "severity": pred_info['severity'],
                "recommendation": pred_info['recommendation']
            },
            "gradcam_overlay": f"data:image/png;base64,{overlay_b64}",
            "disclaimer": "This is an AI screening tool, not a medical diagnosis."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
