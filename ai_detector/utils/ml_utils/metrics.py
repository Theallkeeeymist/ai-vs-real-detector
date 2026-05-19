"""
Evaluation metrics and utilities.
Calculate accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from dataclasses import dataclass
from ai_detector.logging.logger import logger


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str


class MetricsCalculator:
    """
    Calculate evaluation metrics.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        class_names: list = None
    ) -> EvaluationMetrics:
        """
        Calculate all metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            class_names: Class names for report
        """
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        target_names = class_names if class_names else None
        class_report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            classification_report=class_report
        )
    
    @staticmethod
    def get_predictions(model, data_loader, device: str) -> tuple:
        """
        Get predictions from model on entire dataset.
        
        Returns:
            (y_true, y_pred, y_pred_proba)
        """
        all_preds = []
        all_proba = []
        all_labels = []
        
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X = X.to(device)
                
                # Forward pass
                logits = model(X)
                proba = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect
                all_preds.extend(preds.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
                all_labels.extend(y.numpy())
        
        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_proba)
        )