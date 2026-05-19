"""
MLflow utilities for experiment tracking and model versioning.
"""

import mlflow
import mlflow.pytorch
import os
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.constant.training_pipeline import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)
import sys


class MLflowManager:
    """
    Manages MLflow experiment tracking and logging.
    """
    
    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI, 
                 experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
            
            # Set experiment
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            logger.info(f"MLflow experiment: {experiment_name}")
            
        except Exception as e:
            raise AIDetectorException(f"Failed to initialize MLflow", sys)
    
    def start_run(self, run_name: str = None, tags: dict = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add
        """
        try:
            mlflow.start_run(run_name=run_name)
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            logger.info(f"MLflow run started: {run_name}")
            
        except Exception as e:
            raise AIDetectorException(f"Failed to start MLflow run", sys)
    
    def end_run(self):
        """End current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            raise AIDetectorException(f"Failed to end MLflow run", sys)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log metrics for current step/epoch.
        
        Args:
            metrics: Dictionary of metric_name: value
            step: Epoch or step number
        """
        try:
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            raise AIDetectorException(f"Failed to log metrics", sys)
    
    def log_params(self, params: dict):
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of param_name: value
        """
        try:
            for param_name, value in params.items():
                mlflow.log_param(param_name, value)
            logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            raise AIDetectorException(f"Failed to log params", sys)
    
    def log_model(self, model, artifact_path: str = "model"):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model
            artifact_path: Where to save in MLflow
        """
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            raise AIDetectorException(f"Failed to log model", sys)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """
        Log file artifact (CSV, image, etc).
        
        Args:
            artifact_path: Path to file
            artifact_type: "file" or "directory"
        """
        try:
            if artifact_type == "file":
                mlflow.log_artifact(artifact_path)
            else:
                mlflow.log_artifacts(artifact_path)
            logger.info(f"Logged artifact: {artifact_path}")
        except Exception as e:
            raise AIDetectorException(f"Failed to log artifact", sys)
    
    def log_dict(self, dictionary: dict, artifact_file: str = "params.txt"):
        """
        Log dictionary as text artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for artifact
        """
        try:
            import json
            with open(artifact_file, 'w') as f:
                json.dump(dictionary, f, indent=2)
            mlflow.log_artifact(artifact_file)
            logger.info(f"Logged dict to {artifact_file}")
        except Exception as e:
            raise AIDetectorException(f"Failed to log dict", sys)
    
    def get_run_id(self) -> str:
        """Get current run ID."""
        try:
            return mlflow.active_run().info.run_id
        except Exception as e:
            raise AIDetectorException(f"Failed to get run ID", sys)


class MLflowModelRegistry:
    """
    Manage model registry - version and track models.
    """
    
    def __init__(self):
        try:
            self.client = mlflow.tracking.MlflowClient()
            logger.info("MLflow Model Registry initialized")
        except Exception as e:
            raise AIDetectorException(f"Failed to initialize Model Registry", sys)
    
    def register_model(self, model_uri: str, model_name: str) -> str:
        """
        Register model in MLflow Registry.
        
        Args:
            model_uri: URI of model (e.g., "runs:/run_id/model")
            model_name: Name for this model
            
        Returns:
            Model version
        """
        try:
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            logger.info(f"Registered {model_name} (v{version})")
            return version
        except Exception as e:
            raise AIDetectorException(f"Failed to register model", sys)
    
    def set_model_stage(self, model_name: str, version: str, stage: str):
        """
        Transition model to stage (Staging, Production, Archived).
        
        Args:
            model_name: Model name
            version: Model version
            stage: "Staging", "Production", or "Archived"
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"{model_name} v{version} → {stage}")
        except Exception as e:
            raise AIDetectorException(f"Failed to set model stage", sys)
    
    def get_latest_version(self, model_name: str, stage: str = "Production") -> str:
        """
        Get latest model version in a stage.
        
        Args:
            model_name: Model name
            stage: "Staging", "Production", etc.
            
        Returns:
            Latest version number
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            raise AIDetectorException(f"Failed to get latest version", sys)