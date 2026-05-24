"""
Unified inference interface for all 4 models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.constant.training_pipeline import DEVICE
import sys


class ModelPredictor:
    """
    Unified interface for making predictions with all 4 models.
    """
    
    def __init__(self, models: Dict[str, torch.nn.Module], device: str = DEVICE):
        """
        Args:
            models: Dict of loaded models {"model_1": model, ...}
            device: "cuda" or "cpu"
        """
        self.models = models
        self.device = device
        self.class_names = ["Real", "AI Generated"]  # Your classes
        logger.info(f"ModelPredictor initialized with {len(models)} models")
    
    def predict_batch(self, 
                     model_name: str,
                     batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for a batch of images using one model.
        
        Args:
            model_name: "model_1", "model_2", "model_3", or "model_4"
            batch: Tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            (logits, probabilities)
            - logits: [batch_size, 2]
            - probabilities: [batch_size, 2] (softmax applied)
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            
            model = self.models[model_name]
            
            with torch.inference_mode():
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1)
            
            return logits.cpu().numpy(), probabilities.cpu().numpy()
            
        except Exception as e:
            raise AIDetectorException(f"Prediction failed for {model_name}", sys)
    
    def predict_all_models(self, batch: torch.Tensor) -> Dict[str, Dict]:
        """
        Make predictions with all 4 models on the same batch.
        Useful for comparison.
        
        Returns:
        {
            "model_1": {"logits": [...], "probabilities": [...], "predictions": [...]},
            "model_2": {...},
            ...
        }
        """
        try:
            results = {}
            
            for model_name in self.models.keys():
                logits, probs = self.predict_batch(model_name, batch)
                predictions = np.argmax(probs, axis=1)
                
                results[model_name] = {
                    "logits": logits,
                    "probabilities": probs,
                    "predictions": predictions,
                    "class_names": [self.class_names[p] for p in predictions]
                }
            
            return results
            
        except Exception as e:
            raise AIDetectorException(f"Multi-model prediction failed", sys)
    
    def get_model_confidence(self, model_name: str, batch: torch.Tensor) -> np.ndarray:
        """
        Get confidence scores for a model's predictions.
        
        Returns:
            Confidence scores [batch_size]
        """
        try:
            _, probs = self.predict_batch(model_name, batch)
            # Confidence = max probability across classes
            confidence = np.max(probs, axis=1)
            return confidence
            
        except Exception as e:
            raise AIDetectorException(f"Failed to get confidence", sys)
    
    def ensemble_prediction(self, batch: torch.Tensor, method: str = "mean") -> Dict:
        """
        Ensemble predictions from all 4 models.
        
        Args:
            batch: Images [batch_size, 3, 224, 224]
            method: "mean", "max", or "voting"
            
        Returns:
            Ensemble prediction with class, confidence, and individual model votes
        """
        try:
            all_results = self.predict_all_models(batch)
            batch_size = batch.shape[0]
            
            if method == "mean":
                # Average probabilities across models
                ensemble_probs = np.mean(
                    [all_results[m]["probabilities"] for m in self.models.keys()],
                    axis=0
                )
            elif method == "max":
                # Take max probability per class
                ensemble_probs = np.maximum.reduce(
                    [all_results[m]["probabilities"] for m in self.models.keys()]
                )
            elif method == "voting":
                # Majority vote
                predictions = np.array(
                    [all_results[m]["predictions"] for m in self.models.keys()]
                )
                ensemble_probs = np.zeros((batch_size, 2))
                for i in range(batch_size):
                    unique, counts = np.unique(predictions[:, i], return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        ensemble_probs[i, cls] = cnt / len(self.models)
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            
            ensemble_predictions = np.argmax(ensemble_probs, axis=1)
            ensemble_confidence = np.max(ensemble_probs, axis=1)
            
            return {
                "ensemble_predictions": ensemble_predictions,
                "ensemble_confidence": ensemble_confidence,
                "ensemble_class_names": [self.class_names[p] for p in ensemble_predictions],
                "individual_models": all_results
            }
            
        except Exception as e:
            raise AIDetectorException(f"Ensemble prediction failed: {str(e)}", sys)