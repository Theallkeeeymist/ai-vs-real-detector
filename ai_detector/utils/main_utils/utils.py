"""
Utility functions used across the pipeline.
Handles common tasks like saving/loading objects, YAML operations, etc.
"""

import pickle
import yaml
import os
import numpy as np
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
import sys

def save_object(file_path: str, obj: object) -> None:
    """Saves a python (model or preprocessor, etc) to a pickle file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        raise AIDetectorException(f"Failed to save object to {file_path}", sys)

def load_object(file_path: str) -> object:
    """Loads a python object from a pickle file"""

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise AIDetectorException(f"Failed to load object from {file_path}", sys)
    
def read_yaml(file_path: str) -> dict:
    """Reads a YAML file and returns the contents as a dictionary."""

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        logger.info(f"YAML file loaded from {file_path}")
        return content
    except Exception as e:
        raise AIDetectorException(f"Failed to read YAML file: {file_path}", sys)

def write_yaml(file_path: str, content: dict) -> None:
    """Writes a dictionary to a YAML file."""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(content, f, default_flow_style=False)
        logger.info(f"YAML file written to {file_path}")
    except Exception as e:
        raise AIDetectorException(f"Failed to write YAML file: {file_path}", sys)
    
def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """Saves a numpy array to a .npy file."""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
        logger.info(f"Saved numpy array to {file_path}")
    except Exception as e:
        raise AIDetectorException(f"Failed to save numpy array to {file_path}", sys)
    
def load_numpy_array(file_path: str)->np.ndarray:
    """Loads a Numpy array from a .npy file."""

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Numpy file not found: {file_path}")
    
        array = np.load(file_path)
        logger.info(f"Loaded numpy array from {file_path}")
        return array
    except Exception as e:
        raise AIDetectorException(f"Failed to load numpy array from {file_path}", sys)