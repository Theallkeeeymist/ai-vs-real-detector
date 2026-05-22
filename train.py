"""
Main entry point to train all 4 models from scratch.
"""

import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.pipeline.training_pipeline import TrainingPipeline
from ai_detector.utils.gpu_utils import setup_gpu_memory, get_device, log_gpu_memory_status


def main():
    try:
        logger.info("Initializing GPU memory management...")
        setup_gpu_memory()
        device = get_device()
        log_gpu_memory_status()
        
        logger.info("Starting training...")
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline()
        
        logger.info("\n\nTRAINING COMPLETED!")
        logger.info("Models saved to artifacts/[timestamp]/model_trainer/")
        log_gpu_memory_status()
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()