import sys
from ai_detector.logging.logger import logger

class AIDetectorException(Exception):
    """Custom exception for AI Detector Pipeline"""

    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message

        _, _, exc_tb = error_detail.exc_info()
        self.lineno = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None

    def __str__(self):
        return (
            f"Error in [{self.file_name}] at line [{self.lineno}]: {self.error_message}"
        )
    

    def log_error(self):
        logger.error(str(self))