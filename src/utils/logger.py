"""
Logging utility for consistent logging across the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "skin_detection",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file. If None, only console logging.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to also log to console
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger('training', 'logs/train.log')
        >>> logger.info('Training started')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "skin_detection") -> logging.Logger:
    """
    Get an existing logger or create a basic one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set up basic console logging
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
    
    return logger


def create_experiment_log(experiment_name: str) -> logging.Logger:
    """
    Create a logger for a specific experiment with timestamped log file.
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        Logger configured with experiment-specific log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=experiment_name,
        log_file=log_file,
        level=logging.DEBUG
    )


if __name__ == "__main__":
    # Test logging
    logger = setup_logger('test', 'logs/test.log')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print("Logger test complete. Check logs/test.log")
