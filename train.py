"""
Main entry point to train all 4 models from scratch.
"""

import sys
from ai_detector.logging.logger import logger
from ai_detector.exception.exception import AIDetectorException
from ai_detector.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        logger.info("Starting training...")
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline()
        
        logger.info("\n\nTRAINING COMPLETED!")
        logger.info("Models saved to artifacts/[timestamp]/model_trainer/")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()