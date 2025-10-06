import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise ValueError("Invalid image data")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode numpy array image to base64 string."""
    try:
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise

def validate_image_size(image: np.ndarray, max_size: Tuple[int, int] = (1024, 1024)) -> bool:
    """Validate image size constraints."""
    height, width = image.shape[:2]
    max_height, max_width = max_size
    
    if height > max_height or width > max_width:
        logger.warning(f"Image size {width}x{height} exceeds maximum {max_width}x{max_height}")
        return False
    
    if height < 50 or width < 50:
        logger.warning(f"Image size {width}x{height} is too small")
        return False
    
    return True

def resize_image_if_needed(image: np.ndarray, max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Resize image if it exceeds maximum dimensions."""
    height, width = image.shape[:2]
    max_height, max_width = max_size
    
    if height > max_height or width > max_width:
        # Calculate scaling factor
        scale = min(max_height / height, max_width / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return image