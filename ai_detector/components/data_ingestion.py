"""DataIngestion Component - Load images exactly like the notebook
Uses ImageFolder + random_split() for reporductibility"""

import os, sys
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import DataIngestionConfig
from ai_detector.entity.artifact_entity import DataIngestionArtifact
from ai_detector.constant.training_pipeline import RANDOM_SEED

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
            logger.info("DataIngestion Initialized")
        except Exception as e:
            raise AIDetectorException(e, sys)
        
    def load_dataset_with_split(self, dataset_path: str, transform: transforms.Compose,
                                 train_ratio: float=0.7, val_ratio: float=0.15,
                                 test_ratio: float=0.15):
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            df = datasets.ImageFolder(root=dataset_path, transform=transform)

            class_to_idx = df.class_to_idx
            logger.info(f"Classes: {class_to_idx}")

            torch.manual_seed(RANDOM_SEED)
            total_size = len(df)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size  # FIXED: Ensure sum equals total

            logger.info(f"Total: {total_size} | Train: {train_size} | Val: {val_size} | Test: {test_size}")

            train_ds, val_ds, test_ds = random_split(df, [train_size, val_size, test_size])

            logger.info(f"Dataset split complete")

            return train_ds, val_ds, test_ds, class_to_idx
        except Exception as e:
            raise AIDetectorException(f"Failed to load dataset from {dataset_path}", sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Startinng DataIngestion")

            default_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])

            train_ds, val_ds, test_ds, class_labels = self.load_dataset_with_split(self.config.dataset_dir, default_transform,
                                                                                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
            
            artifact = DataIngestionArtifact(
                train_ds, val_ds, test_ds, class_labels, len(train_ds) + len(val_ds) + len(test_ds)
            )

            logger.info("DataIngestion Completed")
            return artifact
        except Exception as e:
            raise AIDetectorException(e, sys)