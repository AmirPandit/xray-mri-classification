import argparse
import json
import time
from pathlib import Path
import logging
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import settings
from src.data.dataloader import create_data_loaders
from models.transfer_model import TransferLearningWithGradCAM
from models.cnn_model import CNNWithGradCAM
from models.utils import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Complete model training pipeline in pure Python."""
    
    def __init__(self, data_dir: Path, models_dir: Path, results_dir: Path):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def setup_data(self, batch_size: int = 32):
        """Setup data loaders for training."""
        logger.info("Setting up data loaders...")
        
        train_dir = self.data_dir / 'processed' / 'train'
        val_dir = self.data_dir / 'processed' / 'val'
        
        if not train_dir.exists():
            raise ValueError(f"Training data not found at {train_dir}")
        
        train_loader, val_loader, class_to_idx = create_data_loaders(
            train_dir, val_dir, batch_size=batch_size
        )
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Classes: {class_to_idx}")
        
        return train_loader, val_loader, class_to_idx
    
    def create_model(self, model_name: str, num_classes: int):
        """Create model instance."""
        logger.info(f"Creating model: {model_name}")
        
        if model_name in ['mobilenet_v2', 'resnet50']:
            model = TransferLearningWithGradCAM(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=True
            )
        elif model_name == 'cnn':
            model = CNNWithGradCAM(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {model_name}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def setup_training(self, model, learning_rate: float):
        """Setup training components."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        return criterion, optimizer, scheduler
    
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, 
                   num_epochs: int, num_classes: int):
        """Train the model."""
        logger.info("Starting model training...")
        
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            num_classes=num_classes
        )
        
        start_time = time.time()
        history = trainer.train(num_epochs, self.models_dir)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history, trainer, training_time
    
    def plot_training_history(self, history, model_name: str):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{model_name} - Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_name}_training_history.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved for {model_name}")
    
    def save_training_report(self, history, model_name: str, training_time: float, 
                           class_to_idx: dict):
        """Save training report."""
        report = {
            'model_name': model_name,
            'training_time_seconds': training_time,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc']),
            'class_mapping': class_to_idx,
            'training_history': history,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save JSON report
        report_path = self.results_dir / f'{model_name}_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save history as CSV
        history_df = pd.DataFrame(history)
        history_path = self.results_dir / f'{model_name}_training_history.csv'
        history_df.to_csv(history_path, index=False)
        
        logger.info(f"Training report saved for {model_name}")
        
        return report
    
    def run_training(self, model_name: str, batch_size: int, learning_rate: float, 
                    num_epochs: int):
        """Run complete training pipeline."""
        logger.info(f"Starting training pipeline for {model_name}")
        
        try:
            # Setup data
            train_loader, val_loader, class_to_idx = self.setup_data(batch_size)
            num_classes = len(class_to_idx)
            
            # Create model
            model = self.create_model(model_name, num_classes)
            
            # Setup training
            criterion, optimizer, scheduler = self.setup_training(model, learning_rate)
            
            # Train model
            history, trainer, training_time = self.train_model(
                model, train_loader, val_loader, criterion, optimizer, num_epochs, num_classes
            )
            
            # Create plots
            self.plot_training_history(history, model_name)
            
            # Save report
            report = self.save_training_report(history, model_name, training_time, class_to_idx)
            
            # Print summary
            self.print_training_summary(report, model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def print_training_summary(self, report: dict, model_name: str):
        """Print training summary to console."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Training time: {report['training_time_seconds']:.2f} seconds")
        print(f"Final training accuracy: {report['final_train_acc']:.2f}%")
        print(f"Final validation accuracy: {report['final_val_acc']:.2f}%")
        print(f"Best validation accuracy: {report['best_val_acc']:.2f}%")
        print(f"Number of classes: {len(report['class_mapping'])}")
        print(f"Classes: {list(report['class_mapping'].keys())}")
        print(f"Results saved to: {self.results_dir}")
        
        # Recommendations based on results
        best_val_acc = report['best_val_acc']
        if best_val_acc < 70:
            print("\n💡 Recommendations:")
            print("   - Consider using more training data")
            print("   - Try different model architecture")
            print("   - Adjust hyperparameters (learning rate, batch size)")
            print("   - Increase data augmentation")
        elif best_val_acc < 85:
            print("\n💡 Good results! Potential improvements:")
            print("   - Fine-tune hyperparameters")
            print("   - Try ensemble methods")
            print("   - Add more sophisticated augmentation")
        else:
            print("\n🎉 Excellent results! Model is ready for deployment.")

def main():
    """Main function for training script."""
    parser = argparse.ArgumentParser(description='Train medical image classification models')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                       choices=['mobilenet_v2', 'resnet50', 'cnn'],
                       help='Model architecture to train')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Path to save models')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    pipeline = ModelTrainingPipeline(
        data_dir=Path(args.data_dir),
        models_dir=Path(args.models_dir),
        results_dir=Path(args.results_dir)
    )
    
    success = pipeline.run_training(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs
    )
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
        exit(1)

if __name__ == "__main__":
    main()