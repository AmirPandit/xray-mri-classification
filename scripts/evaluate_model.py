import argparse
import json
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import logging
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import torch
from torch.utils.data import DataLoader

from src.config.settings import settings
from src.data.dataloader import MedicalImageDataset
from models.transfer_model import TransferLearningWithGradCAM
from models.cnn_model import CNNWithGradCAM
from models.utils import load_model
from torchvision import transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation in pure Python."""
    
    def __init__(self, data_dir: Path, models_dir: Path, results_dir: Path):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    

    def load_test_data(self, batch_size: int = 32):
        """Load test dataset."""
        logger.info("Loading test data...")
        
        test_dir = self.data_dir / 'processed' / 'test'
        if not test_dir.exists():
            raise ValueError(f"Test data not found at {test_dir}")
        
        # Add transforms
        eval_transforms = transforms.Compose([
            transforms.Resize((224, 224)),   # or your model's expected input size
            transforms.ToTensor(),           # PIL -> Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # standard for ImageNet models
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        test_dataset = MedicalImageDataset(test_dir, is_training=False, transform=eval_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Classes: {test_dataset.class_to_idx}")
        
        return test_loader, test_dataset

    
    def load_trained_model(self, model_name: str, num_classes: int):
        """Load trained model."""
        logger.info(f"Loading trained model: {model_name}")
        
        model_path = self.models_dir / 'best_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if model_name in ['mobilenet_v2', 'resnet50']:
            model_class = TransferLearningWithGradCAM
        else:
            model_class = CNNWithGradCAM
        
        model = load_model(
            model_path=model_path,
            model_class=model_class,
            model_name=model_name,
            num_classes=num_classes,
            device=self.device
        )
        
        return model
    
    def evaluate_model(self, model, test_loader):
        """Comprehensive model evaluation."""
        logger.info("Evaluating model...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                
                if self.device.type == 'cuda':
                    start_time.record()
                
                start_cpu = time.time() if self.device.type == 'cpu' else None
                
                outputs = model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                if self.device.type == 'cuda':
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time) / 1000
                else:
                    inference_time = time.time() - start_cpu
                
                inference_times.append(inference_time)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities),
            'inference_times': inference_times
        }
    
    def calculate_metrics(self, targets, predictions, probabilities, class_names):
        """Calculate comprehensive performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision_macro': precision_score(targets, predictions, average='macro'),
            'precision_weighted': precision_score(targets, predictions, average='weighted'),
            'recall_macro': recall_score(targets, predictions, average='macro'),
            'recall_weighted': recall_score(targets, predictions, average='weighted'),
            'f1_macro': f1_score(targets, predictions, average='macro'),
            'f1_weighted': f1_score(targets, predictions, average='weighted'),
        }
        
        # ROC AUC
        if len(class_names) == 2:
            metrics['roc_auc'] = roc_auc_score(targets, probabilities[:, 1])
        else:
            metrics['roc_auc_ovr'] = roc_auc_score(targets, probabilities, multi_class='ovr')
            metrics['roc_auc_ovo'] = roc_auc_score(targets, probabilities, multi_class='ovo')
        
        # Per-class metrics
        class_report = classification_report(targets, predictions, 
                                           target_names=class_names, output_dict=True)
        metrics['per_class'] = class_report
        
        return metrics
    
    def create_confusion_matrix(self, targets, predictions, class_names, model_name: str):
        """Create and save confusion matrix."""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_name}_confusion_matrix.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Normalized Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_name}_confusion_matrix_normalized.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def create_roc_curves(self, targets, probabilities, class_names, model_name: str):
        """Create ROC curves."""
        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(self.results_dir / f'{model_name}_roc_curve.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Multi-class
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve((targets == i).astype(int), probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                
                axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
                axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'ROC Curve - {class_name}')
                axes[i].legend(loc="lower right")
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'{model_name}_roc_curves.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def generate_evaluation_report(self, metrics: dict, model_name: str, 
                                 evaluation_results: dict, class_names: list):
        """Generate comprehensive evaluation report."""
        report = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_set_size': len(evaluation_results['targets']),
            'overall_metrics': {k: v for k, v in metrics.items() if k != 'per_class'},
            'inference_stats': {
                'mean_time': np.mean(evaluation_results['inference_times']),
                'std_time': np.std(evaluation_results['inference_times']),
                'total_time': sum(evaluation_results['inference_times'])
            },
            'class_names': class_names,
            'per_class_metrics': metrics['per_class']
        }
        
        # Save JSON report
        report_path = self.results_dir / f'{model_name}_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([{**report['overall_metrics'], **report['inference_stats']}])
        metrics_df.to_csv(self.results_dir / f'{model_name}_metrics.csv', index=False)
        
        return report
    
    def print_evaluation_summary(self, report: dict, model_name: str):
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Test set size: {report['test_set_size']}")
        print(f"Accuracy: {report['overall_metrics']['accuracy']:.4f}")
        print(f"F1-Score (Weighted): {report['overall_metrics']['f1_weighted']:.4f}")
        
        if 'roc_auc' in report['overall_metrics']:
            print(f"ROC AUC: {report['overall_metrics']['roc_auc']:.4f}")
        else:
            print(f"ROC AUC (OvR): {report['overall_metrics']['roc_auc_ovr']:.4f}")
        
        print(f"Average inference time: {report['inference_stats']['mean_time']:.4f}s")
        print(f"Results saved to: {self.results_dir}")
        
        # Performance assessment
        accuracy = report['overall_metrics']['accuracy']
        if accuracy < 0.7:
            print("\n⚠️  Performance below expected. Consider:")
            print("   - More training data")
            print("   - Different model architecture")
            print("   - Hyperparameter tuning")
        elif accuracy < 0.85:
            print("\n✅ Good performance! Ready for deployment with monitoring.")
        else:
            print("\n🎉 Excellent performance! Ready for production deployment.")
    
    def run_evaluation(self, model_name: str, batch_size: int = 32):
        """Run complete evaluation pipeline."""
        logger.info(f"Starting evaluation for {model_name}")
        
        try:
            # Load data and model
            test_loader, test_dataset = self.load_test_data(batch_size)
            model = self.load_trained_model(model_name, len(test_dataset.class_to_idx))
            
            # Evaluate model
            evaluation_results = self.evaluate_model(model, test_loader)
            
            # Calculate metrics
            class_names = list(test_dataset.class_to_idx.keys())
            metrics = self.calculate_metrics(
                evaluation_results['targets'],
                evaluation_results['predictions'], 
                evaluation_results['probabilities'],
                class_names
            )
            
            # Create visualizations
            self.create_confusion_matrix(
                evaluation_results['targets'],
                evaluation_results['predictions'],
                class_names,
                model_name
            )
            
            self.create_roc_curves(
                evaluation_results['targets'],
                evaluation_results['probabilities'], 
                class_names,
                model_name
            )
            
            # Generate report
            report = self.generate_evaluation_report(
                metrics, model_name, evaluation_results, class_names
            )
            
            # Print summary
            self.print_evaluation_summary(report, model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False

def main():
    """Main function for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate trained medical image classification models')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                       choices=['mobilenet_v2', 'resnet50', 'cnn'],
                       help='Model architecture to evaluate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Path to models directory')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        data_dir=Path(args.data_dir),
        models_dir=Path(args.models_dir),
        results_dir=Path(args.results_dir)
    )
    
    success = evaluator.run_evaluation(
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    if success:
        logger.info("Evaluation completed successfully!")
    else:
        logger.error("Evaluation failed!")
        exit(1)

if __name__ == "__main__":
    main()