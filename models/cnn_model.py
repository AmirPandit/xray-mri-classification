import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleCNN(nn.Module):
    """Custom CNN model for medical image classification."""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate / 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps for Grad-CAM."""
        return self.features(x)

class CNNWithGradCAM(SimpleCNN):
    """CNN model with Grad-CAM support."""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__(num_classes, dropout_rate)
        
        # Store the last convolutional feature maps for Grad-CAM
        self.last_conv_output = None
        
        # Hook to capture feature maps
        def hook_fn(module, input, output):
            self.last_conv_output = output
            
        # Register hook on the last convolutional layer
        self.features[-5].register_forward_hook(hook_fn)  # Last conv layer before dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_conv_output = None
        return super().forward(x)