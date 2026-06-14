import logging
import os
import sys
from datetime import datetime

def setup_logging(script_name: str):
    os.makedirs("logs", exist_ok=True)
    # Remove .py extension if present for cleaner log names
    base_name = os.path.basename(script_name).replace('.py', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{base_name}_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
