import os
import zipfile
import tarfile
import requests
import kaggle
from pathlib import Path
import pandas as pd
import shutil
from typing import Optional
import logging

from src.config.settings import settings
from src.config.datasets import get_dataset_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Handles downloading and setting up medical image datasets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_slug: str, dataset_name: str) -> Path:
        """Download dataset from Kaggle using kaggle API."""
        try:
            import kaggle
        except ImportError:
            raise ImportError("Kaggle API not installed. Run: pip install kaggle")
        
        # Set Kaggle credentials path
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        dataset_path = self.raw_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        logger.info(f"Downloading {dataset_slug} from Kaggle...")
        
        try:
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=dataset_path,
                unzip=True
            )
            logger.info(f"Dataset downloaded to {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            # Alternative: Manual download instructions
            self._print_manual_download_instructions(dataset_slug)
            return None
    
    def download_from_url(self, url: str, dataset_name: str) -> Path:
        """Download dataset from direct URL."""
        import urllib.request
        
        dataset_path = self.raw_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        filename = url.split('/')[-1]
        filepath = dataset_path / filename
        
        logger.info(f"Downloading {url} to {filepath}...")
        
        try:
            urllib.request.urlretrieve(url, filepath)
            logger.info("Download completed")
            
            # Extract if archive
            if filename.endswith('.zip'):
                self._extract_zip(filepath, dataset_path)
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                self._extract_tar(filepath, dataset_path)
                
            return dataset_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def _extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract zip file."""
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Extraction completed")
    
    def _extract_tar(self, tar_path: Path, extract_to: Path):
        """Extract tar file."""
        logger.info(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        logger.info("Extraction completed")
    
    def _print_manual_download_instructions(self, dataset_slug: str):
        """Print instructions for manual download."""
        logger.info("\n" + "="*50)
        logger.info("MANUAL DOWNLOAD REQUIRED")
        logger.info("="*50)
        logger.info(f"Please download the dataset manually:")
        logger.info(f"1. Go to: https://www.kaggle.com/{dataset_slug}")
        logger.info(f"2. Download the dataset")
        logger.info(f"3. Extract it to: {self.raw_dir}")
        logger.info("="*50)
    
    def organize_chest_xray_dataset(self, dataset_path: Path):
        """Organize the chest x-ray pneumonia dataset."""
        logger.info("Organizing Chest X-Ray dataset...")
        
        # Expected structure for chest x-ray dataset
        expected_dirs = ['train', 'test', 'val']
        
        for split in expected_dirs:
            split_path = dataset_path / split
            if split_path.exists():
                logger.info(f"Found {split} directory")
            else:
                logger.warning(f"Missing {split} directory")
        
        return dataset_path
    
    def organize_covid_dataset(self, dataset_path: Path):
        """Organize COVID-19 radiography dataset."""
        logger.info("Organizing COVID-19 dataset...")
        
        # This dataset typically has all images in one folder with a CSV
        csv_files = list(dataset_path.glob('*.csv'))
        if csv_files:
            logger.info(f"Found CSV file: {csv_files[0]}")
        
        return dataset_path
    
    def organize_brain_mri_dataset(self, dataset_path: Path):
        """Organize brain MRI tumor classification dataset."""
        logger.info("Organizing Brain MRI dataset...")
        
        # Expected structure
        training_dir = dataset_path / 'Training'
        testing_dir = dataset_path / 'Testing'
        
        if training_dir.exists() and testing_dir.exists():
            logger.info("Brain MRI dataset structure looks good")
        else:
            logger.warning("Unexpected Brain MRI dataset structure")
        
        return dataset_path

def main():
    """Main function to download and setup datasets."""
    downloader = DatasetDownloader(settings.DATA_DIR)
    
    # Define datasets to download
    datasets_to_download = [
        {
            'name': 'chest_xray_pneumonia',
            'type': 'kaggle',
            'slug': 'paultimothymooney/chest-xray-pneumonia',
            'organize_func': downloader.organize_chest_xray_dataset
        },
        {
            'name': 'covid_chest_xray', 
            'type': 'kaggle',
            'slug': 'tawsifurrahman/covid19-radiography-database',
            'organize_func': downloader.organize_covid_dataset
        },
        {
            'name': 'brain_mri_tumor',
            'type': 'kaggle', 
            'slug': 'sartajbhuvaji/brain-tumor-classification-mri',
            'organize_func': downloader.organize_brain_mri_dataset
        }
    ]
    
    print("Available datasets to download:")
    for i, dataset in enumerate(datasets_to_download, 1):
        print(f"{i}. {dataset['name']} - {dataset['slug']}")
    
    choice = input("\nEnter dataset number to download (or 'all' for all): ").strip()
    
    if choice.lower() == 'all':
        datasets = datasets_to_download
    else:
        try:
            idx = int(choice) - 1
            datasets = [datasets_to_download[idx]]
        except (ValueError, IndexError):
            print("Invalid choice")
            return
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Downloading: {dataset['name']}")
        print(f"{'='*50}")
        
        if dataset['type'] == 'kaggle':
            dataset_path = downloader.download_kaggle_dataset(
                dataset['slug'], 
                dataset['name']
            )
        else:
            # For URL-based downloads
            dataset_path = downloader.download_from_url(
                dataset['url'],
                dataset['name']
            )
        
        if dataset_path and dataset['organize_func']:
            dataset['organize_func'](dataset_path)
        
        print(f"Dataset ready at: {dataset_path}")

if __name__ == "__main__":
    main()