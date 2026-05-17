"""
Configuration classes that are passed to each component.
These define HOW each component should behave.
"""

import os
from datetime import datetime
from ai_detector.constant import training_pipeline

class TrainingPipelineConfig:
    """Contains all configuration for the training pipeline.
    Generates artifact directories with timestamps."""

    def __init__(self, timestamp: str=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        self.timestamp = timestamp
        self.pipeline_name = training_pipeline.PIPELINE_NAME

        self.artifact_dir = os.path.join(
            training_pipeline.ARTIFACT_DIR,
            timestamp
        )

        self.model_dir = training_pipeline.MODELS_DIR

class DataIngestionConfig:
    """COnfiguration for data ingestion component
    Defines where to load data from"""

    def __init__(self, pipeline_config: TrainingPipelineConfig):
        self.dataset_dir = training_pipeline.DATASET_DIR
        self.ai_generated_dir = training_pipeline.AI_GENERATED_DIR
        self.real_dataset_dir = training_pipeline.REAL_DATASET_DIR

        self.image_categories = training_pipeline.IMAGE_CATEGORIES
        self.data_manifest_file = training_pipeline.DATA_MANIFEST_FILE

        self.class_labels = training_pipeline.CLASS_LABELS

        self.data_ingestion_dir = os.path.join(
            pipeline_config.artifact_dir,
            "data_ingestion"
        )
        os.makedirs(self.data_ingestion_dir, exist_ok=True)

class DataValidationConfig:
    """
    Configuration for DataValidation Component
    Defines validation rules
    """

    def __init__(self, pipeline_config: TrainingPipelineConfig):
        self.image_size = training_pipeline.IMAGE_SIZE

        self.data_validation_dir = os.path.join(
            pipeline_config.artifact_dir,
            "data_validation"
        )
        os.makedirs(self.data_validation_dir, exist_ok=True)

        self.validation_report_path = os.path.join(
            self.data_validation_dir,
            "validation_report.txt"
        )

class DataTransformationConfig:
    """
    Configuration for DataTransformation Component
    Defines how to preprocess Image
    """

    def __init__(self, pipeline_config: TrainingPipelineConfig):
        self.image_size = training_pipeline.IMAGE_SIZE
        self.batch_size = training_pipeline.BATCH_SIZE
        self.num_workers = training_pipeline.NUM_WORKERS

        self.data_transformation_dir = os.path.join(
            pipeline_config.artifact_dir,
            "data_transformation"
        )
        os.makedirs(self.data_transformation_dir, exist_ok=True)

class ModelTrainingConfig:
    """
    Configuration for ModelTrainer Component
    Defines hyperparameters for training
    """

    def __init__(self, pipeline_config: TrainingPipelineConfig):
        self.devices = training_pipeline.DEVICE
        self.learning_rate = training_pipeline.LEARNING_RATE
        self.epochs = training_pipeline.EPOCHS
        self.batch_size = training_pipeline.BATCH_SIZE

        self.custom_cnn_config = {
            "input_channels": training_pipeline.CUSTOM_CNN_INPUT,
            "hidden_units": training_pipeline.CUSTOM_CNN_HIDDEN,
            "output_classes": training_pipeline.CUSTOM_CNN_OUTPUT,
        }
        
        self.efficientnet_config = {
            "dropout": training_pipeline.EFFICIENTNET_DROPOUT,
            "hidden_units": training_pipeline.EFFICIENTNET_HIDDEN,
        }
        
        self.vit_config = {
            "output_features": training_pipeline.VIT_OUTPUT_FEATURES,
        }
        
        self.hybrid_config = {
            "efficientnet_features": training_pipeline.HYBRID_EFFICIENTNET_FEATURES,
            "vit_features": training_pipeline.HYBRID_VIT_FEATURES,
            "hidden_units": training_pipeline.HYBRID_HIDDEN,
            "dropout": training_pipeline.HYBRID_DROPOUT,
        }

        self.model_trainer_dir = os.path.join(
            pipeline_config.artifact_dir,
            "model_trainer"
        )
        os.makedirs(self.model_trainer_dir, exist_ok=True)