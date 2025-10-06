import torch
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.config.settings import settings
from src.data.preprocess import preprocessor
from models.utils import load_model
from models.transfer_model import TransferLearningModel
from models.gradcam import apply_gradcam
from src.api.utils import decode_base64_image, encode_image_to_base64, validate_image_size, resize_image_if_needed

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling model predictions."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.model_version = "1.0.0"
        self.class_names = settings.CLASS_NAMES
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load the trained model."""
        try:
            if model_path is None:
                model_path = settings.MODELS_DIR / "best_model.pth"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model based on configuration
            if settings.MODEL_NAME in ["resnet50", "mobilenet_v2"]:
                self.model = load_model(
                    model_path=model_path,
                    model_class=TransferLearningModel,
                    model_name=settings.MODEL_NAME,
                    num_classes=settings.NUM_CLASSES,
                    device=self.device
                )
            else:
                from models.cnn_model import SimpleCNN
                self.model = load_model(
                    model_path=model_path,
                    model_class=SimpleCNN,
                    num_classes=settings.NUM_CLASSES,
                    device=self.device
                )
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
            return False
    
    def predict(self, image_data: str, include_heatmap: bool = True) -> Dict:
        """Make prediction on a single image."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Decode and validate image
            original_image = decode_base64_image(image_data)
            
            if not validate_image_size(original_image):
                original_image = resize_image_if_needed(original_image)
            
            # Preprocess for model
            processed_image = preprocessor.preprocess_uploaded_image(
                cv2.imencode('.jpg', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))[1].tobytes()
            )
            
            # Add batch dimension
            input_tensor = processed_image.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get class name and confidence
            predicted_class = self.class_names[predicted.item()]
            confidence_value = confidence.item()
            
            # Get all class probabilities
            all_probs = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
            
            # Generate heatmap if requested
            heatmap_data = None
            if include_heatmap:
                try:
                    heatmap_image = apply_gradcam(
                        self.model, input_tensor, original_image, predicted.item()
                    )
                    heatmap_data = encode_image_to_base64(heatmap_image)
                except Exception as e:
                    logger.warning(f"Could not generate heatmap: {str(e)}")
            
            inference_time = time.time() - start_time
            
            # Log prediction
            logger.info(
                f"Prediction: {predicted_class} "
                f"(confidence: {confidence_value:.4f}, "
                f"time: {inference_time:.4f}s)"
            )
            
            return {
                "prediction": predicted_class,
                "confidence": confidence_value,
                "all_predictions": all_probs,
                "heatmap_data": heatmap_data,
                "model_version": self.model_version,
                "inference_time": inference_time
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(
        self, 
        images_data: List[str], 
        include_heatmaps: bool = False
    ) -> List[Dict]:
        """Make predictions on multiple images."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        predictions = []
        
        for i, image_data in enumerate(images_data):
            try:
                prediction = self.predict(image_data, include_heatmap=include_heatmaps)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                # Add error entry
                predictions.append({
                    "prediction": "error",
                    "confidence": 0.0,
                    "all_predictions": {},
                    "heatmap_data": None,
                    "model_version": self.model_version,
                    "inference_time": 0.0,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Batch prediction completed: {len(predictions)} images in {total_time:.4f}s")
        
        return predictions

# Global prediction service instance
prediction_service = PredictionService()