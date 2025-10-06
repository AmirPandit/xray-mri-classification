import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
from models.transfer_model import TransferLearningModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        num_classes: int
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs: int, save_path: Path) -> Dict[str, List[float]]:
        """Full training loop."""
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}/{epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_path / 'best_model.pth')
                logger.info(f'New best model saved with val_acc: {val_acc:.2f}%')
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pth'
                self.save_model(checkpoint_path)
        
        # Save final model
        self.save_model(save_path / 'final_model.pth')
        
        return self.history
    
    def save_model(self, path: Path):
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'num_classes': self.num_classes
        }, path)

def load_model(
    model_path: Path,
    model_class: nn.Module,
    model_name: str = "mobilenet_v2",
    num_classes: int = 4,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model
        if model_name in ["resnet50", "mobilenet_v2"]:
            model = TransferLearningModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=False
            )
        else:
            model = model_class(num_classes=num_classes)
        
        state_dict = checkpoint['model_state_dict']

        # Remove "model." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[6:] if k.startswith("model.") else k
            new_state_dict[key] = v

        # Handle classifier layer mismatches for Sequential classifiers
        classifier_layer = model.classifier[-1]  # last Linear
        if isinstance(classifier_layer, nn.Linear):
            keys_to_delete = []
            for k, v in new_state_dict.items():
                if "classifier" in k:
                    if v.shape != classifier_layer.weight.shape:
                        print(f"Ignoring mismatched key: {k}, shape {v.shape}")
                        keys_to_delete.append(k)
            for k in keys_to_delete:
                del new_state_dict[k]

        # Load remaining weights
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise
