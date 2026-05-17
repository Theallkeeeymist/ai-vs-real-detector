"""
Data Classes that represent outputs from each pipeline component.
These are "artifacts" - the deliverables from each step
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class DataIngestionArtifact:
    """Contain paths to train/val/test data split"""
    train_image_paths: List[str]
    val_image_paths: List[str]
    test_image_paths: List[str]

    class_labels: Dict[str, int]
    total_images: int

@dataclass
class DataValidationArtifact:
    """Confirms all images exist, are readable, and have correct dimensions, etc"""

    is_valid: bool
    invalid_images: List[str]
    validation_report: str
    validation_report_path: str

@dataclass
class DataTransformationArtifact:
    """Contains all normalized/augmentated image data ready for model training"""

    transformed_train_data_path: str # Processed train image
    transformed_test_data_path: str  # Processed test image
    transformed_val_data_path: str   # Processed val image

    transformed_config_path: str     # Path to saved transform pipeline (pickle)

@dataclass
class ModelArtifact:
    """Output from a single model after training/loading.
    Represents one of the 4 models
    """

    model_name: str
    model_path: str

    train_accuracy: float=None
    train_f1: float=None
    train_precision: float=None
    train_recall: float=None

    val_accuracy: float=None
    val_f1: float=None
    val_precision: float=None
    val_recall: float=None

    test_accuracy: float=None
    test_f1: float=None
    test_precision: float=None
    test_recall: float=None

@dataclass
class ModelEvaluationArtifact:
    """Contains comparison metrics for all 4 mdoels."""

    mdoel_artifacts: List[ModelArtifact]

    best_model_name: str
    best_model_accuracy: float

    comparison_report_path: str
    comparison_matrices_path: str