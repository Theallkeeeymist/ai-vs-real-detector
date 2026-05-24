from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np

class PredictionResult(BaseModel):
    model_name: str
    prediction: int
    class_name: str
    confidence: float=Field(..., ge=0, le=1)
    probabilities: Dict[str, float]

    class config:
        json_schema_extra = {
            "example": {
                "model_name": "model_1",
                "prediction": 1,
                "class_name": "AI Generated",
                "confidence": 0.95,
                "probabilities": {
                    "Real": 0.05,
                    "AI Generated": 0.95
                }
            }
        }

class SingleImagePrediction(BaseModel):
    image_id: str
    individual_predictions: Dict[str, PredictionResult]
    ensemble_prediction: int
    ensemble_confidence: float=Field(..., ge=0, le=1)
    ensemble_class: str
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "test_1",
                "individual_predictions": {
                    "model_1": {
                        "model_name": "model_1",
                        "prediction": 1,
                        "class_name": "AI Generated",
                        "confidence": 0.95,
                        "probabilities": {
                            "Real": 0.05,
                            "AI Generated": 0.95
                        }
                    }
                },
                "ensemble_prediction": 1,
                "ensemble_confidence": 0.93,
                "ensemble_class": "AI Generated",
                "timestamp": "2026-05-22T01:55:00Z"
            }
        }

class BatchPrediction(BaseModel):
    num_images: int
    predicions: List[SingleImagePrediction]
    processing_time_seconds: float

    class Config:
        json_schema_extra = {
            "example": {
                "num_images": 3,
                "predictions": [],
                "processing_time_seconds": 2.5
            }
        }

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    parameters: int

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "model_1",
                "model_type": "Custom CNN (TinyVGG)",
                "parameters": 5234892
            }
        }
 
 
class SystemStatus(BaseModel):
    """System status information."""
    status: str
    models_loaded: List[str]
    device: str
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ready",
                "models_loaded": ["model_1", "model_2", "model_3", "model_4"],
                "device": "cuda",
                "gpu_available": True,
                "gpu_memory_gb": 4.0
            }
        }
 
 
class ComparisonResult(BaseModel):
    """Model comparison metrics."""
    model_name: str
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "model_1",
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.94,
                "f1_score": 0.92
            }
        }
 
 
class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "All 4 models loaded and ready"
            }
        }
 
 
class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    code: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid Image",
                "detail": "Image must be PNG or JPG",
                "code": 400
            }
        }
 