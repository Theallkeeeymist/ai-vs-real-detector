"""
FastAPI application for AI vs Real Image Detection.
Serves all 4 models with REST endpoints.
Fixed image preprocessing for all models.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
from datetime import datetime
import time
import torchvision
import torchvision.transforms as transforms

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.utils.ml_utils.model_loader import ModelLoader
from ai_detector.utils.gpu_utils import get_gpu_memory_info, log_gpu_memory_status
from ai_detector.api.schemas import (
    SingleImagePrediction,
    BatchPrediction,
    SystemStatus,
    HealthCheck,
    ErrorResponse,
    PredictionResult,
    ModelInfo
)
from ai_detector.constant.training_pipeline import DEVICE

# Create FastAPI app
app = FastAPI(
    title="AI vs Real Image Detector",
    description="Multi-Model API for detecting AI-generated vs real images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}
device = DEVICE
class_names = ["Real", "AI Generated"]

# Transform for all models (will be applied to raw PIL image)
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ViT specific transform
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
vit_transform = vit_weights.transforms()


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    global models, device
    try:
        logger.info("🚀 Starting FastAPI application...")
        logger.info(f"Using device: {device}")
        logger.info("Loading all 4 models...")
        
        loader = ModelLoader(device=device)
        models = loader.load_all_models()
        
        logger.info("✅ All models loaded successfully!")
        logger.info("🌐 FastAPI server ready at http://localhost:8000")
        logger.info("📚 Docs at http://localhost:8000/docs")
        log_gpu_memory_status()
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FastAPI application...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preprocess_image_for_model(image: Image.Image, model_name: str) -> torch.Tensor:
    """
    Preprocess image for specific model.
    Different models need different transforms.
    """
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        
        if model_name in ["model_1", "model_2"]:
            # Simple transform: just resize and to tensor
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        elif model_name == "model_3":
            # ViT needs specific normalization
            image_tensor = vit_transform(image)
        elif model_name == "model_4":
            # Hybrid also uses ViT transforms
            image_tensor = vit_transform(image)
        else:
            # Default
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        return image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise


def predict_single_model(image_tensor: torch.Tensor, model_name: str) -> tuple:
    """
    Get prediction from a single model.
    Returns: (prediction, confidence, probabilities)
    """
    try:
        model = models[model_name]
        
        with torch.inference_mode():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
        
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.max(probs).item()
        probs_array = probs.cpu().numpy()[0]
        
        return prediction, confidence, probs_array
        
    except Exception as e:
        logger.error(f"Prediction failed for {model_name}: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "message": "AI vs Real Image Detector API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return HealthCheck(
        status="healthy",
        message="All 4 models loaded and ready"
    )


@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_status():
    """Get system status and GPU info."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    gpu_info = get_gpu_memory_info()
    
    return SystemStatus(
        status="ready",
        models_loaded=list(models.keys()),
        device=device,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_gb=gpu_info["total_gb"] if gpu_info else None
    )


@app.post("/predict/single", response_model=SingleImagePrediction, tags=["Prediction"])
async def predict_single_image(
    file: UploadFile = File(..., description="Image file (PNG or JPG)")
):
    """
    Make predictions on a single image using all 4 models.
    Returns individual predictions and ensemble result.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        start_time = time.time()
        
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Image must be PNG or JPG"
            )
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"Processing image: {file.filename}")
        
        # Get predictions from all models
        individual_predictions = {}
        all_predictions = []
        
        for model_name in sorted(models.keys()):
            logger.info(f"  Predicting with {model_name}...")
            
            # Preprocess image for this model
            image_tensor = preprocess_image_for_model(image, model_name)
            
            # Get prediction
            prediction, confidence, probs = predict_single_model(image_tensor, model_name)
            
            individual_predictions[model_name] = PredictionResult(
                model_name=model_name,
                prediction=int(prediction),
                class_name=class_names[prediction],
                confidence=float(confidence),
                probabilities={
                    "Real": float(probs[0]),
                    "AI Generated": float(probs[1])
                }
            )
            
            all_predictions.append(prediction)
        
        # Ensemble: majority vote
        ensemble_prediction = max(set(all_predictions), key=all_predictions.count)
        ensemble_confidence = all_predictions.count(ensemble_prediction) / len(all_predictions)
        
        processing_time = time.time() - start_time
        logger.info(f"✓ Processing complete in {processing_time:.2f}s")
        
        return SingleImagePrediction(
            image_id=file.filename,
            individual_predictions=individual_predictions,
            ensemble_prediction=int(ensemble_prediction),
            ensemble_confidence=float(ensemble_confidence),
            ensemble_class=class_names[ensemble_prediction],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Prediction"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple image files")
):
    """
    Make predictions on multiple images.
    Returns predictions for all images.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    try:
        start_time = time.time()
        predictions = []
        
        for file in files:
            # Validate file type
            if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                logger.warning(f"Skipping {file.filename}: Invalid format")
                continue
            
            # Read image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Get predictions from all models
            individual_predictions = {}
            all_predictions = []
            
            for model_name in sorted(models.keys()):
                # Preprocess image for this model
                image_tensor = preprocess_image_for_model(image, model_name)
                
                # Get prediction
                prediction, confidence, probs = predict_single_model(image_tensor, model_name)
                
                individual_predictions[model_name] = PredictionResult(
                    model_name=model_name,
                    prediction=int(prediction),
                    class_name=class_names[prediction],
                    confidence=float(confidence),
                    probabilities={
                        "Real": float(probs[0]),
                        "AI Generated": float(probs[1])
                    }
                )
                
                all_predictions.append(prediction)
            
            # Ensemble: majority vote
            ensemble_prediction = max(set(all_predictions), key=all_predictions.count)
            ensemble_confidence = all_predictions.count(ensemble_prediction) / len(all_predictions)
            
            predictions.append(SingleImagePrediction(
                image_id=file.filename,
                individual_predictions=individual_predictions,
                ensemble_prediction=int(ensemble_prediction),
                ensemble_confidence=float(ensemble_confidence),
                ensemble_class=class_names[ensemble_prediction],
                timestamp=datetime.utcnow().isoformat()
            ))
        
        processing_time = time.time() - start_time
        
        return BatchPrediction(
            num_images=len(predictions),
            predictions=predictions,
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict/compare", tags=["Prediction"])
async def predict_and_compare(
    file: UploadFile = File(..., description="Image file for comparison")
):
    """
    Predict and get detailed comparison of all 4 models.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get predictions from all models
        comparison = {
            "image_id": file.filename,
            "models": {},
            "best_model": None,
            "agreement": 0
        }
        
        all_predictions = []
        
        for model_name in sorted(models.keys()):
            # Preprocess image for this model
            image_tensor = preprocess_image_for_model(image, model_name)
            
            # Get prediction
            prediction, confidence, probs = predict_single_model(image_tensor, model_name)
            
            comparison["models"][model_name] = {
                "prediction": int(prediction),
                "class": class_names[prediction],
                "confidence": float(confidence),
                "probabilities": {
                    "Real": float(probs[0]),
                    "AI Generated": float(probs[1])
                }
            }
            
            all_predictions.append(prediction)
        
        # Calculate agreement
        agreement = sum(1 for p in all_predictions if p == all_predictions[0]) / len(all_predictions)
        comparison["agreement"] = agreement
        
        # Find best model (highest confidence)
        best_model = max(
            comparison["models"].items(),
            key=lambda x: x[1]["confidence"]
        )
        comparison["best_model"] = best_model[0]
        comparison["best_confidence"] = best_model[1]["confidence"]
        
        return comparison
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@app.get("/models", response_model=list[ModelInfo], tags=["Info"])
async def get_models():
    """Get information about all loaded models."""
    return [
        ModelInfo(
            model_name="model_1",
            model_type="Custom CNN (TinyVGG)",
            parameters=5234892
        ),
        ModelInfo(
            model_name="model_2",
            model_type="EfficientNet B0",
            parameters=4083587
        ),
        ModelInfo(
            model_name="model_3",
            model_type="Vision Transformer B-16",
            parameters=86569984
        ),
        ModelInfo(
            model_name="model_4",
            model_type="Hybrid (EfficientNet + ViT)",
            parameters=90653571
        )
    ]


@app.get("/info", tags=["Info"])
async def get_info():
    """Get detailed API information."""
    return {
        "name": "AI vs Real Image Detector",
        "version": "1.0.0",
        "description": "Detects AI-generated images vs real photos using 4 models",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch",
            "model_comparison": "/predict/compare",
            "models_info": "/models"
        },
        "models": {
            "model_1": "Custom CNN - Fast, lightweight",
            "model_2": "EfficientNet B0 - Balanced",
            "model_3": "Vision Transformer - Most accurate",
            "model_4": "Hybrid - Combines ViT + EfficientNet"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )