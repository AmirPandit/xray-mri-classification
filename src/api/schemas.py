from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    image_data: str = Field(..., description="Base64 encoded image data")
    include_heatmap: bool = Field(True, description="Whether to include Grad-CAM heatmap")

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    all_predictions: Dict[str, float] = Field(..., description="Confidences for all classes")
    heatmap_data: Optional[str] = Field(None, description="Base64 encoded heatmap image")
    model_version: str = Field(..., description="Model version used for prediction")
    inference_time: float = Field(..., description="Inference time in seconds")

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    images: List[str] = Field(..., description="List of base64 encoded images")
    include_heatmaps: bool = Field(False, description="Whether to include heatmaps")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_time: float = Field(..., description="Total processing time in seconds")

class PredictionHistory(BaseModel):
    """Schema for prediction history entry."""
    id: str
    timestamp: datetime
    prediction: str
    confidence: float
    image_size: str

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: datetime 

class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    """Schema for user registration."""
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str

class UserResponse(UserBase):
    """Schema for user response (without password)."""
    id: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    """Schema for token payload data."""
    username: Optional[str] = None
    user_id: Optional[str] = None

class LogoutResponse(BaseModel):
    """Schema for logout response."""
    message: str
    success: bool
