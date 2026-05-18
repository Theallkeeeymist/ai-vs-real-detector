"""
Unified model estimator for inference.
Wraps ModelLoader + ModelPredictor.
"""

import torch
from typing import Dict, Tuple
import numpy as np
from torchvision import transforms

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.utils.ml_utils.model_loader import ModelLoader
from ai_detector.utils.ml_utils.predictor import ModelPredictor
from ai_detector.constant.training_pipeline import DEVICE
import sys


class NetworkModelEstimator:
    """
    Main estimator that handles loading models + making predictions.
    User-facing interface.
    """
    
    def __init__(self, device: str = DEVICE):
        try:
            logger.info("Initializing NetworkModelEstimator...")
            
            # Load all models
            loader = ModelLoader(device=device)
            self.models = loader.load_all_models()
            
            # Create predictor
            self.predictor = ModelPredictor(self.models, device=device)
            self.device = device
            
            logger.info("✓ NetworkModelEstimator ready")
            
        except Exception as e:
            raise AIDetectorException(f"Failed to initialize estimator", sys)
    
    def predict(self, 
                images: torch.Tensor,
                model_name: str = "all") -> Dict:
        """
        Make predictions.
        
        Args:
            images: Tensor [batch_size, 3, 224, 224]
            model_name: "model_1", "model_2", "model_3", "model_4", or "all"
            
        Returns:
            Predictions from specified model(s)
        """
        try:
            images = images.to(self.device)
            
            if model_name == "all":
                return self.predictor.predict_all_models(images)
            elif model_name == "ensemble":
                return self.predictor.ensemble_prediction(images)
            else:
                logits, probs = self.predictor.predict_batch(model_name, images)
                return {
                    model_name: {
                        "logits": logits,
                        "probabilities": probs,
                        "predictions": np.argmax(probs, axis=1)
                    }
                }
                
        except Exception as e:
            raise AIDetectorException(f"Prediction failed", sys)