# API package
"""
Skin Disease Detection - API Package

FastAPI-based REST API for inference.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from .main import app

__all__ = ['app']