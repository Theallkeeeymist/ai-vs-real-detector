"""
DataTransformation Component - Fixed
Applies different transforms to train (with augmentation) vs val/test.
"""

import sys
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import DataTransformationConfig
from ai_detector.entity.artifact_entity import (
    DataTransformationArtifact, 
    DataIngestionArtifact,
    DataValidationArtifact
)
from ai_detector.constant.training_pipeline import BATCH_SIZE


class TransformWrapper(Dataset):
    """
    Wrapper to apply transforms to a Subset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)


class DataTransformation:
    """
    Creates DataLoaders with proper train augmentation.
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
    def get_train_transform() -> transforms.Compose:
        """
        Training transform WITH augmentation.
        Prevents overfitting and helps with generalization.
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def get_val_test_transform() -> transforms.Compose:
        """
        Validation/Test transform - NO augmentation.
        Only resize and normalize.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def get_model_3_train_transform() -> transforms.Compose:
        """ViT training with augmentation."""
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize((224, 224)),
            # Then apply ViT normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_model_3_val_test_transform() -> transforms.Compose:
        """ViT validation/test - NO augmentation."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def create_dataloaders(self) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Create DataLoaders with proper transforms per split.
        """
        try:
            logger.info("Creating DataLoaders with transforms per split...")
            
            dataloaders = {}
            
            # Model 1 & 2: Standard transforms
            logger.info("Creating loaders for model_1 and model_2...")
            
            train_transform = self.get_train_transform()
            val_test_transform = self.get_val_test_transform()
            
            train_wrapped_1 = TransformWrapper(self.ingestion_artifact.train_dataset, train_transform)
            val_wrapped_1 = TransformWrapper(self.ingestion_artifact.val_dataset, val_test_transform)
            test_wrapped_1 = TransformWrapper(self.ingestion_artifact.test_dataset, val_test_transform)
            
            train_loader_1 = DataLoader(
                train_wrapped_1,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers
            )
            val_loader_1 = DataLoader(
                val_wrapped_1,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
            test_loader_1 = DataLoader(
                test_wrapped_1,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
            
            dataloaders["model_1"] = (train_loader_1, val_loader_1, test_loader_1)
            dataloaders["model_2"] = (train_loader_1, val_loader_1, test_loader_1)  # Same transforms
            
            logger.info(f"✓ Created loaders for model_1 and model_2")
            
            # Model 3 & 4: ViT transforms
            logger.info("Creating loaders for model_3 and model_4...")
            
            vit_train_transform = self.get_model_3_train_transform()
            vit_val_test_transform = self.get_model_3_val_test_transform()
            
            train_wrapped_3 = TransformWrapper(self.ingestion_artifact.train_dataset, vit_train_transform)
            val_wrapped_3 = TransformWrapper(self.ingestion_artifact.val_dataset, vit_val_test_transform)
            test_wrapped_3 = TransformWrapper(self.ingestion_artifact.test_dataset, vit_val_test_transform)
            
            train_loader_3 = DataLoader(
                train_wrapped_3,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers
            )
            val_loader_3 = DataLoader(
                val_wrapped_3,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
            test_loader_3 = DataLoader(
                test_wrapped_3,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
            
            dataloaders["model_3"] = (train_loader_3, val_loader_3, test_loader_3)
            dataloaders["model_4"] = (train_loader_3, val_loader_3, test_loader_3)  # Same as ViT
            
            logger.info(f"✓ Created loaders for model_3 and model_4")
            
            return dataloaders
            
        except Exception as e:
            raise AIDetectorException(f"Failed to create dataloaders", sys)
    
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
            
            artifact = DataTransformationArtifact(
                dataloaders=dataloaders,
                batch_size=self.config.batch_size,
                image_size=self.config.image_size
            )
            
            logger.info("DataTransformation completed - No data leakage! (Hopefully LMAO)")
            return artifact
            
        except Exception as e:
            raise AIDetectorException(e, sys)