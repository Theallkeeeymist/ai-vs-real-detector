"""
Start the FastAPI server.
"""

import uvicorn
import sys
import torchvision
from ai_detector.logging.logger import logger


if __name__ == "__main__":
    logger.info("Starting AI vs Real Image Detector API...")
    logger.info("Swagger Docs: http://localhost:8000/docs")
    logger.info("ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(
        "ai_detector.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )