"""DataValidation"""

import sys, os

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import DataValidationConfig
from ai_detector.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact

class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        try:
            self.config = config
            self.ingestion_artifact = ingestion_artifact
            logger.info("DataValidation Initialized")
        except Exception as e:
            raise AIDetectorException(e, sys)
        
    def validate_datasets(self) -> bool:
        try:
            logger.info("Validating datasets...")

            train_size = len(self.ingestion_artifact.train_dataset)
            test_size = len(self.ingestion_artifact.test_dataset)
            val_size = len(self.ingestion_artifact.val_dataset)

            assert train_size>0, "Train dataset is empty"
            assert test_size>0, "Test dataset is empty"
            assert val_size>0, "Validation dataset is empty"

            logger.info(f"Validation Passed: Train={train_size}, val={val_size}, test={test_size}")
            return True
        except Exception as e:
            logger.error(f"Data Validation Failed: {e}")
            raise AIDetectorException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting Data Validation")
            is_valid = self.validate_datasets()

            artifact = DataValidationArtifact(
                is_valid=is_valid,
                invalid_images=[],
                validation_report=f"Validated: {self.ingestion_artifact.total_images}"
            )

            logger.info("DataValidation Complete")
            return artifact
        except Exception as e:
            raise AIDetectorException(e, sys)
        