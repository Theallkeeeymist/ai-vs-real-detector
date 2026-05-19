"""
Evaluate all 4 models on test set.
Creates comparison report and visualizations.
"""

import torch
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelEvaluationConfig
)
from ai_detector.components.data_ingestion import DataIngestion
from ai_detector.components.data_validation import DataValidation
from ai_detector.components.data_transformation import DataTransformation
from ai_detector.components.model_evaluator import ModelEvaluator


def main():
    try:
        logger.info("Starting model evaluation...")
        
        # Initialize configs
        pipeline_config = TrainingPipelineConfig()
        
        # Step 1: Data Ingestion
        logger.info("Loading data...")
        data_config = DataIngestionConfig(pipeline_config)
        data_ingestion = DataIngestion(data_config)
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Step 2: Data Validation
        logger.info("Validating data...")
        val_config = DataValidationConfig(pipeline_config)
        data_validation = DataValidation(ingestion_artifact, val_config)
        validation_artifact = data_validation.initiate_data_validation()
        
        # Step 3: Data Transformation
        logger.info("Creating dataloaders...")
        trans_config = DataTransformationConfig(pipeline_config)
        data_transformation = DataTransformation(
            ingestion_artifact,
            validation_artifact,
            trans_config
        )
        transformation_artifact = data_transformation.initiate_data_transformation()
        
        # Step 4: Model Evaluation
        logger.info("Evaluating models...")
        eval_config = ModelEvaluationConfig(pipeline_config)
        evaluator = ModelEvaluator(transformation_artifact, eval_config)
        evaluation_artifact = evaluator.initiate_model_evaluation()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"\nBest Model: {evaluation_artifact.best_model_name}")
        logger.info(f"Best Accuracy: {evaluation_artifact.best_model_accuracy:.4f}")
        logger.info(f"\nReport: {evaluation_artifact.comparison_report_path}")
        logger.info(f"Visualizations: {evaluation_artifact.confusion_matrices_path}")
        
        # Display comparison table
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info(evaluation_artifact.comparison_report.to_string())
        
        return evaluation_artifact
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()