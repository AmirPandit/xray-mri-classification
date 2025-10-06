import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import logging
from sklearn.model_selection import train_test_split

from src.config.settings import settings
from src.config.datasets import get_dataset_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDatasetProcessor:
    """Process real medical datasets for the project."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_chest_xray_pneumonia(self):
        """Setup chest x-ray pneumonia dataset."""
        logger.info("Setting up Chest X-Ray Pneumonia dataset...")
        
        dataset_path = self.raw_dir / 'chest_xray_pneumonia'
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return False
        
        # Create processed structure
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = self.processed_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images to processed directory
        splits = ['train', 'test', 'val']
        for split in splits:
            split_path = dataset_path / split
            if split_path.exists():
                for class_name in ['NORMAL', 'PNEUMONIA']:
                    class_path = split_path / class_name
                    if class_path.exists():
                        images = list(class_path.glob('*.jpeg'))
                        logger.info(f"Found {len(images)} images in {split}/{class_name}")
                        
                        for img_path in images:
                            dest_path = self.processed_dir / split / class_name / img_path.name
                            shutil.copy2(img_path, dest_path)
        
        logger.info("Chest X-Ray dataset setup completed")
        return True
    
    def setup_covid_chest_xray(self):
        """Setup COVID-19 chest x-ray dataset."""
        logger.info("Setting up COVID-19 Chest X-Ray dataset...")
        
        dataset_path = self.raw_dir / 'covid_chest_xray'
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return False
        
        # Look for the main directory structure
        covid_dir = None
        for possible_dir in ['COVID-19_Radiography_Dataset', 'COVID-19_Radiography_Dataset/*']:
            possible_path = dataset_path / possible_dir
            if possible_path.exists():
                covid_dir = possible_path
                break
        
        if not covid_dir:
            # Try to find the directory with COVID in name
            for item in dataset_path.iterdir():
                if 'COVID' in item.name and item.is_dir():
                    covid_dir = item
                    break
        
        if not covid_dir:
            logger.error("Could not find COVID dataset directory")
            return False
        
        # Create processed structure
        classes = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        for split in ['train', 'val', 'test']:
            for class_name in classes:
                class_dir = self.processed_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each class
        for class_name in classes:
            class_path = covid_dir / class_name
            if class_path.exists():
                images = list(class_path.glob('*.png'))
                logger.info(f"Found {len(images)} images for {class_name}")
                
                # Split into train/val/test (70/15/15)
                train_imgs, test_imgs = train_test_split(
                    images, test_size=0.3, random_state=42
                )
                val_imgs, test_imgs = train_test_split(
                    test_imgs, test_size=0.5, random_state=42
                )
                
                # Copy images to respective directories
                for img_path in train_imgs:
                    dest_path = self.processed_dir / 'train' / class_name / img_path.name
                    shutil.copy2(img_path, dest_path)
                
                for img_path in val_imgs:
                    dest_path = self.processed_dir / 'val' / class_name / img_path.name
                    shutil.copy2(img_path, dest_path)
                
                for img_path in test_imgs:
                    dest_path = self.processed_dir / 'test' / class_name / img_path.name
                    shutil.copy2(img_path, dest_path)
        
        logger.info("COVID-19 dataset setup completed")
        return True
    
    def setup_brain_mri_tumor(self):
        """Setup brain MRI tumor classification dataset."""
        logger.info("Setting up Brain MRI Tumor dataset...")
        
        dataset_path = self.raw_dir / 'brain_mri_tumor'
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return False
        
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        for split in ['train', 'val', 'test']:
            for class_name in classes:
                class_dir = self.processed_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
        
        class_map = {
            "meningioma_tumor": "meningioma",
            "glioma_tumor": "glioma",
            "pituitary_tumor": "pituitary",
            "no_tumor": "notumor"
        }
        
        training_dir = dataset_path / 'Training'
        if training_dir.exists():
            for original_class, mapped_class in class_map.items():
                class_path = training_dir / original_class
                if class_path.exists():
                    images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
                    logger.info(f"Found {len(images)} training images for {mapped_class}")
                    
                    # Split training into train/val (85/15)
                    train_imgs, val_imgs = train_test_split(images, test_size=0.15, random_state=42)
                    
                    for img_path in train_imgs:
                        dest_path = self.processed_dir / 'train' / mapped_class / img_path.name
                        shutil.copy2(img_path, dest_path)
                    
                    for img_path in val_imgs:
                        dest_path = self.processed_dir / 'val' / mapped_class / img_path.name
                        shutil.copy2(img_path, dest_path)
        
        # Process testing data
        testing_dir = dataset_path / 'Testing'
        if testing_dir.exists():
            for original_class, mapped_class in class_map.items():
                class_path = testing_dir / original_class
                if class_path.exists():
                    images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
                    logger.info(f"Found {len(images)} testing images for {mapped_class}")
                    
                    for img_path in images:
                        dest_path = self.processed_dir / 'test' / mapped_class / img_path.name
                        shutil.copy2(img_path, dest_path)
        
        logger.info("Brain MRI dataset setup completed")
        return True

    def create_dataset_info(self):
        """Create dataset information file."""
        dataset_info = {
            'total_images': 0,
            'classes': {},
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_path = self.processed_dir / split
            if split_path.exists():
                dataset_info['splits'][split] = {}
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        images = list(class_dir.glob('*.*'))
                        count = len(images)
                        dataset_info['splits'][split][class_dir.name] = count
                        dataset_info['total_images'] += count
                        
                        if class_dir.name not in dataset_info['classes']:
                            dataset_info['classes'][class_dir.name] = 0
                        dataset_info['classes'][class_dir.name] += count
        
        # Save dataset info
        import json
        info_path = self.processed_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info("Dataset information:")
        logger.info(f"Total images: {dataset_info['total_images']}")
        logger.info(f"Classes: {list(dataset_info['classes'].keys())}")
        for split, classes in dataset_info['splits'].items():
            logger.info(f"{split}: {sum(classes.values())} images")
        
        return dataset_info

def main():
    """Main function to setup real datasets."""
    processor = RealDatasetProcessor(settings.DATA_DIR)
    
    datasets = {
        '1': ('Chest X-Ray Pneumonia', processor.setup_chest_xray_pneumonia),
        '2': ('COVID-19 Chest X-Ray', processor.setup_covid_chest_xray),
        '3': ('Brain MRI Tumor', processor.setup_brain_mri_tumor)
    }
    
    print("Available datasets to setup:")
    for key, (name, _) in datasets.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter dataset number to setup: ").strip()
    
    if choice in datasets:
        name, setup_func = datasets[choice]
        print(f"\nSetting up {name}...")
        success = setup_func()
        
        if success:
            print("Dataset setup completed successfully!")
            # Create dataset info
            processor.create_dataset_info()
        else:
            print("Dataset setup failed!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()