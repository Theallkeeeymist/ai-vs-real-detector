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
    train_dataset: object        # PyTorch Dataset (from random_split)
    val_dataset: object          # PyTorch Dataset
    test_dataset: object         # PyTorch Dataset
    class_labels: dict           # {"real_dataset": 0, "ai_generated_dataset": 1}
    total_images: int            # Total image count

@dataclass
class DataValidationArtifact:
    """Confirms all images exist, are readable, and have correct dimensions, etc"""

    is_valid: bool
    invalid_images: List[str]
    validation_report: str

@dataclass
class DataTransformationArtifact:
    """Contains all normalized/augmentated image data ready for model training"""

    dataloaders: dict  # {"model_1": (train, val, test), "model_2": ...}
    batch_size: int
    image_size: int

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