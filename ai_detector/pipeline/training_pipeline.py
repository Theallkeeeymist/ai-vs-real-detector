"""
Complete Training Pipeline - Orchestrates all components.
DataIngestion → Validation → Transformation → ModelTraining → Evaluation
"""

import sys
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.components.data_ingestion import DataIngestion
from ai_detector.components.data_validation import DataValidation
from ai_detector.components.data_transformation import DataTransformation
from ai_detector.components.model_trainer import ModelTrainer
from ai_detector.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)


class TrainingPipeline:
    """
    Main pipeline that ties everything together.
    Run once to train all 4 models from scratch.
    """
    
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            logger.info(f"TrainingPipeline initialized")
            logger.info(f"Artifact directory: {self.training_pipeline_config.artifact_dir}")
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def run_data_ingestion(self):
        """
        Step 1: Load data from dataset folders.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("STEP 1: DATA INGESTION")
            logger.info("="*80)
            
            config = DataIngestionConfig(self.training_pipeline_config)
            component = DataIngestion(config)
            artifact = component.initiate_data_ingestion()
            
            logger.info(f"✓ Data Ingestion complete")
            return artifact
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def run_data_validation(self, ingestion_artifact):
        """
        Step 2: Validate data.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("STEP 2: DATA VALIDATION")
            logger.info("="*80)
            
            config = DataValidationConfig(self.training_pipeline_config)
            component = DataValidation(ingestion_artifact, config)
            artifact = component.initiate_data_validation()
            
            if not artifact.is_valid:
                raise Exception("Data validation failed!")
            
            logger.info(f"✓ Data Validation complete")
            return artifact
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def run_data_transformation(self, ingestion_artifact, validation_artifact):
        """
        Step 3: Transform data (create DataLoaders).
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("STEP 3: DATA TRANSFORMATION")
            logger.info("="*80)
            
            config = DataTransformationConfig(self.training_pipeline_config)
            component = DataTransformation(ingestion_artifact, validation_artifact, config)
            artifact = component.initiate_data_transformation()
            
            logger.info(f"✓ Data Transformation complete")
            return artifact
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def run_model_training(self, data_transformation_artifact):
        """
        Step 4: Train all 4 models.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("STEP 4: MODEL TRAINING")
            logger.info("="*80)
            
            config = ModelTrainerConfig(self.training_pipeline_config)
            component = ModelTrainer(data_transformation_artifact, config)
            results = component.initiate_model_trainer()
            
            logger.info(f"✓ Model Training complete")
            return results
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def run_pipeline(self):
        """
        Main method - runs the entire pipeline.
        """
        try:
            logger.info("\n\n")
            logger.info("="*76)
            logger.info(" STARTING COMPLETE TRAINING PIPELINE")
            logger.info("="*76)
            
            # Step 1: Data Ingestion
            ingestion_artifact = self.run_data_ingestion()
            
            # Step 2: Data Validation
            validation_artifact = self.run_data_validation(ingestion_artifact)
            
            # Step 3: Data Transformation
            transformation_artifact = self.run_data_transformation(
                ingestion_artifact, 
                validation_artifact
            )
            
            # Step 4: Model Training
            training_results = self.run_model_training(transformation_artifact)
            
            # Summary
            logger.info("\n\n")
            logger.info("="*76)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*76)
            logger.info(f"\nArtifact directory: {self.training_pipeline_config.artifact_dir}")
            
            return training_results
            
        except Exception as e:
            raise AIDetectorException(f"Pipeline failed", sys)