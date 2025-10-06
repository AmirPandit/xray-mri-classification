import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings configuration."""
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Model configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_v2")
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "2"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    # Training configuration
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    EPOCHS = int(os.getenv("EPOCHS", "10"))
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # Data augmentation
    AUGMENTATION_CONFIG = {
        "rotation_limit": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "horizontal_flip": True,
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2]
    }
    
    # Model classes (example - should match your dataset)
    CLASS_NAMES = ["normal", "abnormal"]  # Update based on your dataset

settings = Settings()