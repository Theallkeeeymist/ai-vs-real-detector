"""
ModelTrainer component - trains all 4 models.
Orchestrates training pipeline for each model.
"""

import torch
import os
from typing import Dict, Tuple
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import ModelTrainerConfig
from ai_detector.entity.artifact_entity import ModelEvaluationArtifact, DataTransformationArtifact
from ai_detector.utils.ml_utils.models import (
    AiOrReal,
    create_efficientnet_model,
    create_vit_model,
    AIHybrid
)
from ai_detector.utils.ml_utils.model_loader import ModelLoader
from ai_detector.components.training_utils import (
    get_metrics_trackers,
    train_step,
    test_step
)
from ai_detector.constant.training_pipeline import (
    CUSTOM_CNN_INPUT,
    CUSTOM_CNN_HIDDEN,
    CUSTOM_CNN_OUTPUT,
    RANDOM_SEED
)
import sys


class ModelTrainer:
    """
    Trains all 4 models from scratch.
    Saves state_dict to .pth files.
    """
    
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.config = config
            self.device = config.device
            
            # Get metric trackers
            self.acc_fn, self.f1_score, self.precision_score, self.recall_score = \
                get_metrics_trackers(self.device)
            
            # Loss function (from your notebook)
            self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
            
            logger.info("ModelTrainer initialized")
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def train_model_1(self) -> Dict:
        """
        Train Model 1 (Custom CNN).
        Exactly like your notebook.
        """
        try:
            logger.info("="*80)
            logger.info("Training Model 1 (Custom CNN)")
            logger.info("="*80)
            
            # Initialize model
            torch.manual_seed(RANDOM_SEED)
            model = AiOrReal(
                input_shape=CUSTOM_CNN_INPUT,
                hidden_units=CUSTOM_CNN_HIDDEN,
                output_shape=CUSTOM_CNN_OUTPUT
            ).to(self.device)
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Get dataloaders
            train_loader, val_loader, test_loader = \
                self.data_transformation_artifact.dataloaders["model_1"]
            
            # Training loop
            metrics_history = {"train": [], "val": []}
            
            for epoch in range(self.config.epochs):
                logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")
                
                # Train
                train_metrics = train_step(
                    model=model,
                    data_loader=train_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    device=self.device
                )
                metrics_history["train"].append(train_metrics)
                
                # Validate
                val_metrics = test_step(
                    model=model,
                    data_loader=val_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    device=self.device
                )
                metrics_history["val"].append(val_metrics)
            
            # Test
            logger.info("\n" + "="*40)
            logger.info("TESTING Model 1")
            logger.info("="*40)
            test_metrics = test_step(
                model=model,
                data_loader=test_loader,
                accuracy_fn=self.acc_fn,
                f1_score=self.f1_score,
                precision=self.precision_score,
                recall=self.recall_score,
                loss_fn=self.loss_fn,
                device=self.device
            )
            
            # Save model
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            model_path = os.path.join(self.config.model_trainer_dir, "model_1_custom_cnn.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Model 1 saved to {model_path}")
            
            return {
                "model_name": "model_1",
                "model_path": model_path,
                "train_metrics": metrics_history["train"][-1],
                "val_metrics": metrics_history["val"][-1],
                "test_metrics": test_metrics,
                "history": metrics_history
            }
            
        except Exception as e:
            raise AIDetectorException(f"Failed to train Model 1", sys)
    
    def train_model_2(self) -> Dict:
        """
        Train Model 2 (EfficientNet B0).
        Exactly like your notebook.
        """
        try:
            logger.info("="*80)
            logger.info("Training Model 2 (EfficientNet B0)")
            logger.info("="*80)
            
            # Initialize model
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
            model = create_efficientnet_model(device=self.device)
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Get dataloaders
            train_loader, val_loader, test_loader = \
                self.data_transformation_artifact.dataloaders["model_2"]
            
            # Training loop
            metrics_history = {"train": [], "val": []}
            
            for epoch in range(self.config.epochs):
                logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")
                
                # Train
                train_metrics = train_step(
                    model=model,
                    data_loader=train_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    device=self.device
                )
                metrics_history["train"].append(train_metrics)
                
                # Validate
                val_metrics = test_step(
                    model=model,
                    data_loader=val_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    device=self.device
                )
                metrics_history["val"].append(val_metrics)
            
            # Test
            logger.info("\n" + "="*40)
            logger.info("TESTING Model 2")
            logger.info("="*40)
            test_metrics = test_step(
                model=model,
                data_loader=test_loader,
                accuracy_fn=self.acc_fn,
                f1_score=self.f1_score,
                precision=self.precision_score,
                recall=self.recall_score,
                loss_fn=self.loss_fn,
                device=self.device
            )
            
            # Save model
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            model_path = os.path.join(self.config.model_trainer_dir, "model_2_efficientnet.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Model 2 saved to {model_path}")
            
            return {
                "model_name": "model_2",
                "model_path": model_path,
                "train_metrics": metrics_history["train"][-1],
                "val_metrics": metrics_history["val"][-1],
                "test_metrics": test_metrics,
                "history": metrics_history
            }
            
        except Exception as e:
            raise AIDetectorException(f"Failed to train Model 2", sys)
    
    def train_model_3(self) -> Dict:
        """
        Train Model 3 (ViT B-16).
        Exactly like your notebook.
        """
        try:
            logger.info("="*80)
            logger.info("Training Model 3 (ViT B-16)")
            logger.info("="*80)
            
            # Initialize model
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
            model = create_vit_model(device=self.device)
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Get dataloaders
            train_loader, val_loader, test_loader = \
                self.data_transformation_artifact.dataloaders["model_3"]
            
            # Training loop
            metrics_history = {"train": [], "val": []}
            
            for epoch in range(self.config.epochs):
                logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")
                
                # Train
                train_metrics = train_step(
                    model=model,
                    data_loader=train_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    device=self.device
                )
                metrics_history["train"].append(train_metrics)
                
                # Validate
                val_metrics = test_step(
                    model=model,
                    data_loader=val_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    device=self.device
                )
                metrics_history["val"].append(val_metrics)
            
            # Test
            logger.info("\n" + "="*40)
            logger.info("TESTING Model 3")
            logger.info("="*40)
            test_metrics = test_step(
                model=model,
                data_loader=test_loader,
                accuracy_fn=self.acc_fn,
                f1_score=self.f1_score,
                precision=self.precision_score,
                recall=self.recall_score,
                loss_fn=self.loss_fn,
                device=self.device
            )
            
            # Save model
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            model_path = os.path.join(self.config.model_trainer_dir, "model_3_vit.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Model 3 saved to {model_path}")
            
            return {
                "model_name": "model_3",
                "model_path": model_path,
                "train_metrics": metrics_history["train"][-1],
                "val_metrics": metrics_history["val"][-1],
                "test_metrics": test_metrics,
                "history": metrics_history
            }
            
        except Exception as e:
            raise AIDetectorException(f"Failed to train Model 3", sys)
    
    def train_model_4(self, model_2, model_3) -> Dict:
        """
        Train Model 4 (Hybrid Fusion).
        Requires Model 2 and Model 3 as components.
        """
        try:
            logger.info("="*80)
            logger.info("Training Model 4 (Hybrid Fusion)")
            logger.info("="*80)
            
            # Initialize hybrid model
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
            model = AIHybrid(model_2, model_3).to(self.device)
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Get dataloaders
            train_loader, val_loader, test_loader = \
                self.data_transformation_artifact.dataloaders["model_4"]
            
            # Training loop
            metrics_history = {"train": [], "val": []}
            
            for epoch in range(self.config.epochs):
                logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")
                
                # Train
                train_metrics = train_step(
                    model=model,
                    data_loader=train_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    device=self.device
                )
                metrics_history["train"].append(train_metrics)
                
                # Validate
                val_metrics = test_step(
                    model=model,
                    data_loader=val_loader,
                    accuracy_fn=self.acc_fn,
                    f1_score=self.f1_score,
                    precision=self.precision_score,
                    recall=self.recall_score,
                    loss_fn=self.loss_fn,
                    device=self.device
                )
                metrics_history["val"].append(val_metrics)
            
            # Test
            logger.info("\n" + "="*40)
            logger.info("TESTING Model 4")
            logger.info("="*40)
            test_metrics = test_step(
                model=model,
                data_loader=test_loader,
                accuracy_fn=self.acc_fn,
                f1_score=self.f1_score,
                precision=self.precision_score,
                recall=self.recall_score,
                loss_fn=self.loss_fn,
                device=self.device
            )
            
            # Save model
            os.makedirs(self.config.model_trainer_dir, exist_ok=True)
            model_path = os.path.join(self.config.model_trainer_dir, "model_4_hybrid.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Model 4 saved to {model_path}")
            
            return {
                "model_name": "model_4",
                "model_path": model_path,
                "train_metrics": metrics_history["train"][-1],
                "val_metrics": metrics_history["val"][-1],
                "test_metrics": test_metrics,
                "history": metrics_history
            }
            
        except Exception as e:
            raise AIDetectorException(f"Failed to train Model 4", sys)
    
    def initiate_model_trainer(self) -> Dict:
        """
        Main method - trains all 4 models sequentially.
        """
        try:
            logger.info("Starting Model Trainer - Training All 4 Models")
            
            results = {}
            
            # Train Model 1
            results["model_1"] = self.train_model_1()
            
            # Train Model 2
            results["model_2"] = self.train_model_2()
            
            # Train Model 3
            results["model_3"] = self.train_model_3()
            
            # Load Model 2 and 3 for Hybrid
            # (Need the actual model objects, not just state dicts)
            model_2 = create_efficientnet_model(device=self.device)
            model_2.load_state_dict(torch.load(results["model_2"]["model_path"]))
            
            model_3 = create_vit_model(device=self.device)
            model_3.load_state_dict(torch.load(results["model_3"]["model_path"]))
            
            # Train Model 4
            results["model_4"] = self.train_model_4(model_2, model_3)
            
            logger.info("\n" + "="*80)
            logger.info("✓ ALL 4 MODELS TRAINED SUCCESSFULLY")
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            raise AIDetectorException(f"Model training failed", sys)