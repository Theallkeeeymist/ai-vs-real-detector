"""
Visualization utilities for evaluation.
Confusion matrices, ROC curves, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict
from ai_detector.logging.logger import logger


class EvaluationVisualizer:
    """
    Create visualizations for model evaluation.
    """
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: list,
        model_name: str,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: [2, 2] confusion matrix
            class_names: ["Real", "AI Generated"]
            model_name: "model_1", etc.
            save_path: Where to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=True
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Saved confusion matrix to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_model_comparison(
        metrics_dict: Dict[str, Dict],
        class_names: list,
        save_dir: str = None
    ) -> plt.Figure:
        """
        Compare metrics across all models.
        
        Args:
            metrics_dict: {
                "model_1": {"accuracy": 0.95, "f1": 0.94, ...},
                "model_2": {...},
                ...
            }
            class_names: ["Real", "AI Generated"]
            save_dir: Where to save figures
        """
        
        model_names = list(metrics_dict.keys())
        metrics = ["accuracy", "precision", "recall", "f1"]
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for model_name in model_names:
            for metric in metrics:
                data[metric].append(metrics_dict[model_name][metric])
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(model_names, data[metric], color=colors)
            
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "model_comparison.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Saved comparison plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_all_confusion_matrices(
        confusion_matrices_dict: Dict[str, np.ndarray],
        class_names: list,
        save_dir: str = None
    ) -> plt.Figure:
        """
        Plot all confusion matrices in one figure.
        
        Args:
            confusion_matrices_dict: {
                "model_1": confusion_matrix,
                "model_2": confusion_matrix,
                ...
            }
        """
        model_names = list(confusion_matrices_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (model_name, cm) in enumerate(confusion_matrices_dict.items()):
            ax = axes[idx]
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar=True
            )
            
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "all_confusion_matrices.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Saved all confusion matrices to {save_path}")
        
        return fig