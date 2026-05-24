"""
DataIngestion Component - Fixed
Proper data splitting with train augmentation to prevent data leakage.
"""

import os
import sys
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split, Subset
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
    
    def load_dataset_with_split(self, dataset_path: str,
                                 train_ratio: float=0.7, val_ratio: float=0.15,
                                 test_ratio: float=0.15):
        """
        Load dataset and create train/val/test splits.
        CRITICAL: Use NO transforms during loading, apply after split.
        """
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            
            # Load with NO transforms first
            df = datasets.ImageFolder(root=dataset_path, transform=None)

            class_to_idx = df.class_to_idx
            logger.info(f"Classes: {class_to_idx}")

            torch.manual_seed(RANDOM_SEED)
            total_size = len(df)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size  # Ensure sum = total

            logger.info(f"Total: {total_size} | Train: {train_size} | Val: {val_size} | Test: {test_size}")

            # Split FIRST (on raw data, no transforms)
            train_indices, val_indices, test_indices = random_split(
                range(total_size),
                [train_size, val_size, test_size]
            )

            # Now convert to actual subsets
            train_ds = Subset(df, train_indices.indices)
            val_ds = Subset(df, val_indices.indices)
            test_ds = Subset(df, test_indices.indices)

            logger.info(f"Dataset split complete - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

            return train_ds, val_ds, test_ds, class_to_idx
            
        except Exception as e:
            raise AIDetectorException(f"Failed to load dataset from {dataset_path}", sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main data ingestion method.
        """
        try:
            logger.info("Starting DataIngestion")

            # Load raw dataset (no transforms)
            train_ds, val_ds, test_ds, class_labels = self.load_dataset_with_split(
                self.config.dataset_dir,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            artifact = DataIngestionArtifact(
                train_ds, val_ds, test_ds, class_labels, len(train_ds) + len(val_ds) + len(test_ds)
            )

            logger.info("Leakage Removed")
            return artifact
            
        except Exception as e:
            raise AIDetectorException(e, sys)