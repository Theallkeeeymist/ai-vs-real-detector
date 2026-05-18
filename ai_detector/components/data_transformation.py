"""
DataTransformation Component - Creates DataLoaders
Handles model-specific transforms.
"""

import sys
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import timm

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import DataTransformationConfig
from ai_detector.entity.artifact_entity import (
    DataTransformationArtifact, 
    DataIngestionArtifact,
    DataValidationArtifact
)
from ai_detector.constant.training_pipeline import BATCH_SIZE


class DataTransformation:
    """
    Creates DataLoaders with model-specific transforms.
    Replicates your notebook's different transform approaches.
    """
    
    def __init__(self,
                 ingestion_artifact: DataIngestionArtifact,
                 validation_artifact: DataValidationArtifact,
                 config: DataTransformationConfig):
        try:
            self.ingestion_artifact = ingestion_artifact
            self.validation_artifact = validation_artifact
            self.config = config
            logger.info("DataTransformation initialized")
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    @staticmethod
    def get_model_1_transform() -> transforms.Compose:
        """Custom CNN transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def get_model_2_transform() -> transforms.Compose:
        """EfficientNet transform """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def get_model_3_transform() -> transforms.Compose:
        """
        ViT-B-16 transform .
        Uses official ViT weights transforms.
        """
        
        weights_2 = torchvision.models.ViT_B_16_Weights.DEFAULT
        vit_transforms = weights_2.transforms()
        
        return vit_transforms
    
    @staticmethod
    def get_model_4_transform() -> transforms.Compose:
        """Hybrid model - same as ViT since it uses ViT backbone"""
        return DataTransformation.get_model_3_transform()
    
    def create_dataloaders(self) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Create DataLoaders for each model with their specific transforms.
        
        Returns dict like:
        {
            "model_1": (train_loader, val_loader, test_loader),
            "model_2": (train_loader, val_loader, test_loader),
            "model_3": (train_loader, val_loader, test_loader),
            "model_4": (train_loader, val_loader, test_loader),
        }
        """
        try:
            logger.info("Creating DataLoaders with model-specific transforms...")
            
            # Get transforms for each model
            transforms_dict = {
                "model_1": self.get_model_1_transform(),
                "model_2": self.get_model_2_transform(),
                "model_3": self.get_model_3_transform(),
                "model_4": self.get_model_4_transform(),
            }
            
            dataloaders = {}
            
            for model_name, transform in transforms_dict.items():
                logger.info(f"Creating loaders for {model_name}...")
                
                # Apply transform to datasets
                train_ds_transformed = self._apply_transform_to_dataset(
                    self.ingestion_artifact.train_dataset, transform
                )
                val_ds_transformed = self._apply_transform_to_dataset(
                    self.ingestion_artifact.val_dataset, transform
                )
                test_ds_transformed = self._apply_transform_to_dataset(
                    self.ingestion_artifact.test_dataset, transform
                )
                
                # Create loaders (like your notebook)
                train_loader = DataLoader(
                    train_ds_transformed,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers
                )
                val_loader = DataLoader(
                    val_ds_transformed,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers
                )
                test_loader = DataLoader(
                    test_ds_transformed,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers
                )
                
                dataloaders[model_name] = (train_loader, val_loader, test_loader)
                logger.info(f"✓ Created loaders for {model_name}")
            
            return dataloaders
            
        except Exception as e:
            raise AIDetectorException(f"Failed to create dataloaders", sys)
    
    def _apply_transform_to_dataset(self, dataset, transform):
        """
        Apply new transform to a dataset.
        Works with Subset objects from random_split().
        """
        # Get the underlying ImageFolder dataset
        if hasattr(dataset, 'dataset'):
            # It's a Subset
            base_dataset = dataset.dataset
            indices = dataset.indices
        else:
            base_dataset = dataset
            indices = list(range(len(dataset)))
        
        # Create a new dataset with the transform
        from torch.utils.data import Subset
        base_dataset.transform = transform
        return Subset(base_dataset, indices)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main method - creates all DataLoaders.
        """
        try:
            logger.info("Starting DataTransformation")
            
            if not self.validation_artifact.is_valid:
                raise Exception("Data validation failed!")
            
            # Create dataloaders
            dataloaders = self.create_dataloaders()
            
            # Save reference
            self.dataloaders = dataloaders
            
            artifact = DataTransformationArtifact(
                dataloaders=dataloaders,
                batch_size=self.config.batch_size,
                image_size=self.config.image_size
            )
            
            logger.info("✓ DataTransformation completed")
            return artifact
            
        except Exception as e:
            raise AIDetectorException(e, sys)