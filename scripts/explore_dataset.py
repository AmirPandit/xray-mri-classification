import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any

from src.config.settings import settings
from src.config.datasets import get_dataset_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetExplorer:
    """Comprehensive dataset exploration in pure Python."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.results_dir = self.data_dir / 'analysis'
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze the directory structure of the dataset."""
        logger.info("Analyzing dataset structure...")
        
        structure = {}
        total_images = 0
        
        if self.raw_dir.exists():
            for dataset_dir in self.raw_dir.iterdir():
                if dataset_dir.is_dir():
                    structure[dataset_dir.name] = {}
                    
                    for split_dir in dataset_dir.iterdir():
                        if split_dir.is_dir():
                            structure[dataset_dir.name][split_dir.name] = {}
                            
                            for class_dir in split_dir.iterdir():
                                if class_dir.is_dir():
                                    images = list(class_dir.glob('*.*'))
                                    image_files = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.dcm']]
                                    count = len(image_files)
                                    structure[dataset_dir.name][split_dir.name][class_dir.name] = count
                                    total_images += count
        
        logger.info(f"Found {total_images} total images across {len(structure)} datasets")
        return structure
    
    def analyze_image_statistics(self, sample_size: int = 100) -> pd.DataFrame:
        """Analyze image dimensions, formats, and basic statistics."""
        logger.info("Analyzing image statistics...")
        
        image_data = []
        datasets_analyzed = 0
        
        for dataset_dir in self.raw_dir.iterdir():
            if dataset_dir.is_dir():
                datasets_analyzed += 1
                logger.info(f"Analyzing dataset: {dataset_dir.name}")
                
                for split_dir in dataset_dir.iterdir():
                    if split_dir.is_dir():
                        for class_dir in split_dir.iterdir():
                            if class_dir.is_dir():
                                images = list(class_dir.glob('*.*'))[:sample_size]
                                for img_path in images:
                                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                        try:
                                            with Image.open(img_path) as img:
                                                image_data.append({
                                                    'dataset': dataset_dir.name,
                                                    'split': split_dir.name,
                                                    'class': class_dir.name,
                                                    'path': str(img_path),
                                                    'format': img.format,
                                                    'width': img.width,
                                                    'height': img.height,
                                                    'mode': img.mode,
                                                    'size_kb': img_path.stat().st_size / 1024
                                                })
                                        except Exception as e:
                                            logger.warning(f"Could not read {img_path}: {e}")
        
        df = pd.DataFrame(image_data)
        if not df.empty:
            logger.info(f"Analyzed {len(df)} images from {datasets_analyzed} datasets")
        else:
            logger.warning("No images found for analysis")
        
        return df
    
    def generate_summary_report(self, structure: Dict, image_stats: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset summary report."""
        logger.info("Generating summary report...")
        
        summary = {
            'total_datasets': len(structure),
            'total_images': 0,
            'datasets': {},
            'issues': []
        }
        
        # Analyze each dataset
        for dataset_name, splits in structure.items():
            dataset_summary = {
                'total_images': 0,
                'splits': {},
                'classes': set(),
                'formats': set()
            }
            
            for split_name, classes in splits.items():
                dataset_summary['splits'][split_name] = {
                    'total_images': sum(classes.values()),
                    'classes': classes
                }
                dataset_summary['total_images'] += sum(classes.values())
                dataset_summary['classes'].update(classes.keys())
            
            summary['total_images'] += dataset_summary['total_images']
            summary['datasets'][dataset_name] = dataset_summary
            
            # Check for common issues
            if dataset_summary['total_images'] == 0:
                summary['issues'].append(f"Dataset '{dataset_name}' has no images")
            
            # Check class imbalance
            for split_name, split_data in dataset_summary['splits'].items():
                if len(split_data['classes']) > 1:
                    counts = list(split_data['classes'].values())
                    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
                    if imbalance_ratio > 3:
                        summary['issues'].append(
                            f"Dataset '{dataset_name}' split '{split_name}' has class imbalance (ratio: {imbalance_ratio:.1f})"
                        )
        
        # Add image statistics to summary
        if not image_stats.empty:
            summary['image_statistics'] = {
                'formats': image_stats['format'].value_counts().to_dict(),
                'modes': image_stats['mode'].value_counts().to_dict(),
                'dimensions': {
                    'min_width': image_stats['width'].min(),
                    'max_width': image_stats['width'].max(),
                    'mean_width': image_stats['width'].mean(),
                    'min_height': image_stats['height'].min(),
                    'max_height': image_stats['height'].max(),
                    'mean_height': image_stats['height'].mean()
                }
            }
        
        return summary
    
    def create_visualizations(self, image_stats: pd.DataFrame, summary: Dict[str, Any]):
        """Create visualization plots for the dataset analysis."""
        logger.info("Creating visualizations...")
        
        if image_stats.empty:
            logger.warning("No data available for visualizations")
            return
        
        # 1. Image dimensions scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(image_stats['width'], image_stats['height'], alpha=0.6)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title('Image Dimensions Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'image_dimensions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Class distribution (for first dataset)
        if summary['datasets']:
            first_dataset = list(summary['datasets'].keys())[0]
            dataset_data = summary['datasets'][first_dataset]
            
            for split_name, split_data in dataset_data['splits'].items():
                if split_data['classes']:
                    plt.figure(figsize=(12, 6))
                    classes = list(split_data['classes'].keys())
                    counts = list(split_data['classes'].values())
                    
                    plt.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
                    plt.xlabel('Classes')
                    plt.ylabel('Number of Images')
                    plt.title(f'Class Distribution - {first_dataset} - {split_name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.results_dir / f'class_distribution_{first_dataset}_{split_name}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
        
        # 3. Image format distribution
        if 'image_statistics' in summary:
            formats = summary['image_statistics']['formats']
            plt.figure(figsize=(10, 6))
            plt.pie(formats.values(), labels=formats.keys(), autopct='%1.1f%%')
            plt.title('Image Format Distribution')
            plt.savefig(self.results_dir / 'image_formats.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.results_dir}")
    
    def generate_recommendations(self, summary: Dict[str, Any]) -> list:
        """Generate recommendations based on dataset analysis."""
        recommendations = []
        
        if summary['total_images'] == 0:
            recommendations.append("❌ No images found. Please add medical images to data/raw/ directory.")
            return recommendations
        
        recommendations.append("✅ Dataset analysis completed successfully!")
        
        # Check dataset size
        if summary['total_images'] < 1000:
            recommendations.extend([
                "📊 Dataset size is relatively small. Consider:",
                "   - Using transfer learning with pretrained models",
                "   - Implementing extensive data augmentation",
                "   - Exploring additional data sources"
            ])
        elif summary['total_images'] < 10000:
            recommendations.extend([
                "📊 Moderate dataset size detected.",
                "   - Transfer learning recommended",
                "   - Moderate data augmentation should help",
                "   - Consider fine-tuning pretrained models"
            ])
        else:
            recommendations.extend([
                "📊 Large dataset size detected - excellent!",
                "   - You can train models from scratch",
                "   - Consider ensemble methods",
                "   - Hyperparameter tuning will be valuable"
            ])
        
        # Check for issues
        if summary['issues']:
            recommendations.append("⚠️  Issues detected:")
            recommendations.extend([f"   - {issue}" for issue in summary['issues']])
        
        # Image dimension recommendations
        if 'image_statistics' in summary:
            stats = summary['image_statistics']['dimensions']
            avg_width = stats['mean_width']
            avg_height = stats['mean_height']
            
            if avg_width != 224 or avg_height != 224:
                recommendations.extend([
                    "📐 Image resizing recommended:",
                    f"   - Current average size: {avg_width:.0f}x{avg_height:.0f}",
                    "   - Recommended size for transfer learning: 224x224",
                    "   - Implement resizing in preprocessing"
                ])
        
        # Next steps
        recommendations.extend([
            "\n🚀 Next steps:",
            "   1. Run preprocessing: python scripts/preprocess_dataset.py",
            "   2. Train model: python scripts/train_model.py",
            "   3. Evaluate results: python scripts/evaluate_model.py"
        ])
        
        return recommendations
    
    def save_report(self, summary: Dict[str, Any], recommendations: list):
        """Save analysis report to files."""
        # Save JSON report
        report_path = self.results_dir / 'dataset_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save recommendations as text
        recommendations_path = self.results_dir / 'recommendations.txt'
        with open(recommendations_path, 'w') as f:
            f.write("\n".join(recommendations))
        
        # Save summary as CSV
        if summary['datasets']:
            summary_data = []
            for dataset_name, dataset_info in summary['datasets'].items():
                for split_name, split_info in dataset_info['splits'].items():
                    for class_name, count in split_info['classes'].items():
                        summary_data.append({
                            'dataset': dataset_name,
                            'split': split_name,
                            'class': class_name,
                            'count': count
                        })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(self.results_dir / 'dataset_summary.csv', index=False)
        
        logger.info(f"Reports saved to {self.results_dir}")
    
    def run_complete_analysis(self):
        """Run complete dataset analysis pipeline."""
        logger.info("Starting comprehensive dataset analysis...")
        
        # Run analysis steps
        structure = self.analyze_dataset_structure()
        image_stats = self.analyze_image_statistics()
        summary = self.generate_summary_report(structure, image_stats)
        
        # Create visualizations
        self.create_visualizations(image_stats, summary)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(summary)
        
        # Save reports
        self.save_report(summary, recommendations)
        
        # Print summary to console
        print("\n" + "="*60)
        print("DATASET ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Total images: {summary['total_images']}")
        
        if summary['datasets']:
            for dataset_name, dataset_info in summary['datasets'].items():
                print(f"\n📁 {dataset_name}:")
                print(f"   Total images: {dataset_info['total_images']}")
                print(f"   Classes: {len(dataset_info['classes'])}")
                for split_name, split_info in dataset_info['splits'].items():
                    print(f"   {split_name}: {split_info['total_images']} images")
        
        print("\n💡 RECOMMENDATIONS:")
        for recommendation in recommendations:
            print(f"   {recommendation}")
        
        print(f"\n📊 Full report saved to: {self.results_dir}")
        logger.info("Dataset analysis completed!")

def main():
    """Main function for dataset exploration."""
    parser = argparse.ArgumentParser(description='Explore and analyze medical image datasets')
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='Path to data directory (default: data)')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of images to sample per class (default: 100)')
    
    args = parser.parse_args()
    
    explorer = DatasetExplorer(Path(args.data_dir))
    explorer.run_complete_analysis()

if __name__ == "__main__":
    main()