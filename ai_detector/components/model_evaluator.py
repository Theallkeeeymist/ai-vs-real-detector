"""
ModelEvaluator - Evaluates all 4 models on test set.
"""

import sys
import os
from typing import Dict
import pandas as pd

from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.entity.config_entity import ModelEvaluationConfig
from ai_detector.utils.ml_utils.mlflow_utils import MLflowManager, MLflowModelRegistry
from ai_detector.entity.artifact_entity import (
    ModelEvaluationArtifact,
    DataTransformationArtifact
)
from ai_detector.utils.ml_utils.model_loader import ModelLoader
from ai_detector.utils.ml_utils.metrics import MetricsCalculator
from ai_detector.utils.ml_utils.visualizations import EvaluationVisualizer
from ai_detector.constant.training_pipeline import DEVICE


class ModelEvaluator:
    """
    Evaluates all 4 models on test set.
    Generates comparison report and visualizations.
    """
    
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 config: ModelEvaluationConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.config = config
            self.device = DEVICE
            self.class_names = ["Real", "AI Generated"]
            
            # Load all models
            logger.info("Loading models for evaluation...")
            loader = ModelLoader(device=self.device)
            self.models = loader.load_all_models()
            
            logger.info("✓ ModelEvaluator initialized")
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def evaluate_model(self, model_name: str) -> Dict:
        """
        Evaluate a single model.
        
        Returns:
        {
            "model_name": "model_1",
            "metrics": EvaluationMetrics,
            "y_true": [...],
            "y_pred": [...],
            "y_pred_proba": [...]
        }
        """
        try:
            logger.info(f"\nEvaluating {model_name}...")
            
            model = self.models[model_name]
            
            # Get test dataloader
            _, _, test_loader = \
                self.data_transformation_artifact.dataloaders[model_name]
            
            # Get predictions on test set
            y_true, y_pred, y_pred_proba = MetricsCalculator.get_predictions(
                model,
                test_loader,
                self.device
            )
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_metrics(
                y_true,
                y_pred,
                y_pred_proba,
                self.class_names
            )
            
            logger.info(f"✓ {model_name} Evaluation Complete")
            logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
            logger.info(f"  Precision: {metrics.precision:.4f}")
            logger.info(f"  Recall: {metrics.recall:.4f}")
            logger.info(f"  F1: {metrics.f1:.4f}")
            if metrics.roc_auc:
                logger.info(f"  ROC-AUC: {metrics.roc_auc:.4f}")
            
            return {
                "model_name": model_name,
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba
            }
            
        except Exception as e:
            raise AIDetectorException(f"Failed to evaluate {model_name}", sys)
    
    def evaluate_all_models(self) -> Dict:
        """
        Evaluate all 4 models.
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("EVALUATING ALL 4 MODELS ON TEST SET")
            logger.info("="*80)
            
            results = {}
            
            for model_name in self.models.keys():
                results[model_name] = self.evaluate_model(model_name)
            
            logger.info("\n" + "="*80)
            logger.info("✓ ALL MODELS EVALUATED")
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            raise AIDetectorException(e, sys)
    
    def generate_comparison_report(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Generate comparison report table.
        
        Returns:
            DataFrame with all metrics for all models
        """
        try:
            logger.info("\nGenerating comparison report...")
            
            report_data = []
            
            for model_name, result in evaluation_results.items():
                metrics = result["metrics"]
                
                report_data.append({
                    "Model": model_name,
                    "Accuracy": f"{metrics.accuracy:.4f}",
                    "Precision": f"{metrics.precision:.4f}",
                    "Recall": f"{metrics.recall:.4f}",
                    "F1-Score": f"{metrics.f1:.4f}",
                    "ROC-AUC": f"{metrics.roc_auc:.4f}" if metrics.roc_auc else "N/A"
                })
            
            df_report = pd.DataFrame(report_data)
            
            # Save to CSV
            report_path = self.config.comparison_report_path
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            df_report.to_csv(report_path, index=False)
            logger.info(f"✓ Saved report to {report_path}")
            
            # Print to console
            logger.info("\n" + "="*80)
            logger.info("MODEL COMPARISON REPORT")
            logger.info("="*80)
            logger.info(df_report.to_string())
            logger.info("="*80)
            
            return df_report
            
        except Exception as e:
            raise AIDetectorException(f"Failed to generate report", sys)
    
    def generate_confusion_matrices(self, evaluation_results: Dict) -> Dict:
        """
        Generate confusion matrices for all models.
        """
        try:
            logger.info("\nGenerating confusion matrices...")
            
            confusion_matrices = {}
            
            for model_name, result in evaluation_results.items():
                cm = result["metrics"].confusion_matrix
                confusion_matrices[model_name] = cm
                
                # Plot individual confusion matrix
                save_path = os.path.join(
                    self.config.confusion_matrices_path,
                    f"{model_name}_confusion_matrix.png"
                )
                EvaluationVisualizer.plot_confusion_matrix(
                    cm,
                    self.class_names,
                    model_name,
                    save_path
                )
            
            # Plot all confusion matrices together
            all_cm_path = os.path.join(
                self.config.confusion_matrices_path,
                "all_confusion_matrices.png"
            )
            EvaluationVisualizer.plot_all_confusion_matrices(
                confusion_matrices,
                self.class_names,
                self.config.confusion_matrices_path
            )
            
            logger.info("✓ Confusion matrices generated")
            
            return confusion_matrices
            
        except Exception as e:
            raise AIDetectorException(f"Failed to generate confusion matrices", sys)
    
    def generate_comparison_plots(self, evaluation_results: Dict):
        """
        Generate comparison plots.
        """
        try:
            logger.info("\nGenerating comparison plots...")
            
            # Prepare metrics dict
            metrics_dict = {}
            for model_name, result in evaluation_results.items():
                metrics = result["metrics"]
                metrics_dict[model_name] = {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "roc_auc": metrics.roc_auc if metrics.roc_auc else 0.0
                }
            
            # Plot comparison
            EvaluationVisualizer.plot_model_comparison(
                metrics_dict,
                self.class_names,
                self.config.confusion_matrices_path
            )
            
            logger.info("✓ Comparison plots generated")
            
        except Exception as e:
            raise AIDetectorException(f"Failed to generate plots", sys)
    
    def generate_detailed_reports(self, evaluation_results: Dict):
        """
        Generate detailed classification reports for each model.
        """
        try:
            logger.info("\nGenerating detailed reports...")
            
            report_dir = self.config.confusion_matrices_path
            os.makedirs(report_dir, exist_ok=True)
            
            for model_name, result in evaluation_results.items():
                metrics = result["metrics"]
                
                report_path = os.path.join(report_dir, f"{model_name}_report.txt")
                
                with open(report_path, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write("SUMMARY METRICS\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Accuracy:  {metrics.accuracy:.4f}\n")
                    f.write(f"Precision: {metrics.precision:.4f}\n")
                    f.write(f"Recall:    {metrics.recall:.4f}\n")
                    f.write(f"F1-Score:  {metrics.f1:.4f}\n")
                    if metrics.roc_auc:
                        f.write(f"ROC-AUC:   {metrics.roc_auc:.4f}\n")
                    
                    f.write("\n\nCLASSIFICATION REPORT\n")
                    f.write("-"*80 + "\n")
                    f.write(metrics.classification_report)
                    
                    f.write("\n\nCONFUSION MATRIX\n")
                    f.write("-"*80 + "\n")
                    f.write(str(metrics.confusion_matrix))
                
                logger.info(f"✓ Saved report for {model_name}")
            
        except Exception as e:
            raise AIDetectorException(f"Failed to generate detailed reports", sys)
    
def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
    """
    Main method - runs complete evaluation with MLflow logging.
    """
    try:
        logger.info("\n" + "="*40)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("="*40)
        
        # Initialize MLflow
        mlflow_manager = MLflowManager()
        mlflow_manager.start_run(
            run_name="model_evaluation",
            tags={
                "phase": "evaluation",
                "dataset": "ai_vs_real",
                "models": "4_models_comparison"
            }
        )
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models()
        
        # Log evaluation metrics for all models
        for model_name, result in evaluation_results.items():
            metrics = result["metrics"]
            
            mlflow_manager.log_metrics({
                f"{model_name}_accuracy": metrics.accuracy,
                f"{model_name}_precision": metrics.precision,
                f"{model_name}_recall": metrics.recall,
                f"{model_name}_f1": metrics.f1,
                f"{model_name}_roc_auc": metrics.roc_auc if metrics.roc_auc else 0.0,
            })
        
        # Generate comparison report
        report_df = self.generate_comparison_report(evaluation_results)
        
        # Log report as artifact
        mlflow_manager.log_artifact(self.config.comparison_report_path)
        
        # Generate confusion matrices
        confusion_matrices = self.generate_confusion_matrices(evaluation_results)
        
        # Log confusion matrix visualizations
        mlflow_manager.log_artifact(self.config.confusion_matrices_path, artifact_type="directory")
        
        # Generate comparison plots
        self.generate_comparison_plots(evaluation_results)
        
        # Generate detailed reports
        self.generate_detailed_reports(evaluation_results)
        
        # Find best model
        best_model_name = None
        best_accuracy = -1
        
        for model_name, result in evaluation_results.items():
            if result["metrics"].accuracy > best_accuracy:
                best_accuracy = result["metrics"].accuracy
                best_model_name = model_name
        
        # Log best model info
        mlflow_manager.log_metrics({
            "best_model_accuracy": best_accuracy
        })
        mlflow_manager.log_params({
            "best_model": best_model_name
        })
        
        logger.info("\n" + "="*40)
        logger.info(f"BEST MODEL: {best_model_name} with Accuracy: {best_accuracy:.4f}")
        logger.info("="*40)
        
        # End MLflow run
        mlflow_manager.end_run()
        
        # Create artifact
        artifact = ModelEvaluationArtifact(
            evaluation_results=evaluation_results,
            comparison_report=report_df,
            confusion_matrices=confusion_matrices,
            best_model_name=best_model_name,
            best_model_accuracy=best_accuracy,
            comparison_report_path=self.config.comparison_report_path,
            confusion_matrices_path=self.config.confusion_matrices_path
        )
        
        return artifact
        
    except Exception as e:
        raise AIDetectorException(e, sys)