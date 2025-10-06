import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any

class TransferLearningModel(nn.Module):
    """Transfer learning model with pretrained backbone."""
    
    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model
        if model_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            in_features = backbone.fc.in_features
            # Remove the final fully connected layer
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
            self.gradcam_layer = self.features[-1]  # Last convolutional layer
            
        elif model_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(pretrained=pretrained)
            in_features = backbone.last_channel  # 1280
            self.features = backbone.features
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),  # global average pooling
                nn.Flatten(),                  # flatten to [batch, 1280]
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
            self.gradcam_layer = self.features[-1]

            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Store the last convolutional feature maps for Grad-CAM
        self.last_conv_output = None
        
        # Register hook for Grad-CAM
        def hook_fn(module, input, output):
            self.last_conv_output = output
            
        self.gradcam_layer.register_forward_hook(hook_fn)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_conv_output = None
        x = self.features(x)
        x = self.classifier(x)
        return x

class TransferLearningWithGradCAM(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower()
        if model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            )
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)  # num_classes = 4
            )
            self.gradcam_layer = list(self.model.features.children())[-1]

            
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            # Replace classifier to match checkpoint
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            # Last conv layer for Grad-CAM
            self.gradcam_layer = self.model.layer4[-1].conv3
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    def forward(self, x):
        return self.model(x)
