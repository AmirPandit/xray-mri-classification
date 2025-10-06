import logging
import logging.config
import sys
from pathlib import Path

def setup_logging():
    """Configure logging for the application."""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
            "file": {
                "level": "DEBUG",
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_dir / "api.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {
                "handlers": ["default", "file"],
                "level": "INFO",
                "propagate": False
            },
            "src": {
                "handlers": ["default", "file"],
                "level": "DEBUG",
                "propagate": False
            },
        }
    }
    
    logging.config.dictConfig(logging_config)