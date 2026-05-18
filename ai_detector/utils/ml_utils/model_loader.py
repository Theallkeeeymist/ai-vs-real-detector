"""
Load your 4 .pth model files and prepare them for inference.
"""

import torch
import os
from typing import Dict
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.utils.ml_utils.models import (
    AiOrReal,
    create_efficientnet_model,
    create_vit_model,
    AIHybrid
)
from ai_detector.constant.training_pipeline import (
    MODEL_CHECKPOINT_1,
    MODEL_CHECKPOINT_2,
    MODEL_CHECKPOINT_3,
    MODEL_CHECKPOINT_4,
    DEVICE,
    CUSTOM_CNN_INPUT,
    CUSTOM_CNN_HIDDEN,
    CUSTOM_CNN_OUTPUT
)
import sys


class ModelLoader:
    """
    Loads all 4 models from .pth files.
    Handles model initialization + state_dict loading.
    """
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.models = {}
        logger.info(f"ModelLoader initialized on device: {device}")
    
    def load_model_1(self) -> torch.nn.Module:
        """
        Load Custom CNN model.
        Initialize architecture, then load state_dict from .pth
        """
        try:
            logger.info(f"Loading Model 1 (Custom CNN) from {MODEL_CHECKPOINT_1}...")
            
            if not os.path.exists(MODEL_CHECKPOINT_1):
                raise FileNotFoundError(f"Model 1 not found: {MODEL_CHECKPOINT_1}")
            
            # Initialize architecture
            model = AiOrReal(
                input_shape=CUSTOM_CNN_INPUT,
                hidden_units=CUSTOM_CNN_HIDDEN,
                output_shape=CUSTOM_CNN_OUTPUT
            ).to(self.device)
            
            # Load state_dict (saved weights)
            state_dict = torch.load(MODEL_CHECKPOINT_1, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.eval()  # Set to evaluation mode
            logger.info("✓ Model 1 loaded successfully")
            return model
            
        except Exception as e:
            raise AIDetectorException(f"Failed to load Model 1", sys)
    
    def load_model_2(self) -> torch.nn.Module:
        """
        Load EfficientNet B0 model.
        Initialize with pretrained weights, then load your fine-tuned state_dict.
        """
        try:
            logger.info(f"Loading Model 2 (EfficientNet B0) from {MODEL_CHECKPOINT_2}...")
            
            if not os.path.exists(MODEL_CHECKPOINT_2):
                raise FileNotFoundError(f"Model 2 not found: {MODEL_CHECKPOINT_2}")
            
            # Initialize with your exact configuration from notebook
            model = create_efficientnet_model(device=self.device)
            
            # Load your fine-tuned weights
            state_dict = torch.load(MODEL_CHECKPOINT_2, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.eval()
            logger.info("✓ Model 2 loaded successfully")
            return model
            
        except Exception as e:
            raise AIDetectorException(f"Failed to load Model 2", sys)
    
    def load_model_3(self) -> torch.nn.Module:
        """
        Load ViT B-16 model.
        Initialize with pretrained weights, then load your fine-tuned state_dict.
        """
        try:
            logger.info(f"Loading Model 3 (ViT B-16) from {MODEL_CHECKPOINT_3}...")
            
            if not os.path.exists(MODEL_CHECKPOINT_3):
                raise FileNotFoundError(f"Model 3 not found: {MODEL_CHECKPOINT_3}")
            
            # Initialize with your exact configuration from notebook
            model = create_vit_model(device=self.device)
            
            # Load your fine-tuned weights
            state_dict = torch.load(MODEL_CHECKPOINT_3, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.eval()
            logger.info("✓ Model 3 loaded successfully")
            return model
            
        except Exception as e:
            raise AIDetectorException(f"Failed to load Model 3", sys)
    
    def load_model_4(self, model_2: torch.nn.Module, model_3: torch.nn.Module) -> torch.nn.Module:
        """
        Load Hybrid Fusion model.
        Needs Model 2 (EfficientNet) and Model 3 (ViT) as components.
        """
        try:
            logger.info(f"Loading Model 4 (Hybrid) from {MODEL_CHECKPOINT_4}...")
            
            if not os.path.exists(MODEL_CHECKPOINT_4):
                raise FileNotFoundError(f"Model 4 not found: {MODEL_CHECKPOINT_4}")
            
            # Initialize Hybrid with the loaded models
            model = AIHybrid(model_2, model_3).to(self.device)
            
            # Load your fine-tuned weights
            state_dict = torch.load(MODEL_CHECKPOINT_4, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.eval()
            logger.info("✓ Model 4 loaded successfully")
            return model
            
        except Exception as e:
            raise AIDetectorException(f"Failed to load Model 4", sys)
    
    def load_all_models(self) -> Dict[str, torch.nn.Module]:
        """
        Load all 4 models.
        Returns dictionary: {"model_1": model, "model_2": model, ...}
        """
        try:
            logger.info("Loading all 4 models...")
            
            model_1 = self.load_model_1()
            model_2 = self.load_model_2()
            model_3 = self.load_model_3()
            model_4 = self.load_model_4(model_2, model_3)
            
            self.models = {
                "model_1": model_1,
                "model_2": model_2,
                "model_3": model_3,
                "model_4": model_4
            }
            
            logger.info("✓ All 4 models loaded successfully")
            return self.models
            
        except Exception as e:
            raise AIDetectorException(e, sys)