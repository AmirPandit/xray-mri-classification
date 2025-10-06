# src/config/datasets.py
"""Configuration for different medical image datasets."""
from pathlib import Path
from typing import Dict, Any

class DatasetConfig:
    """Configuration for different medical image datasets."""
    
    # Chest X-Ray Datasets
    CHEST_XRAY_KAGGLE = {
        'name': 'chest_xray_kaggle',
        'url': 'https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia',
        'expected_structure': {
            'train': ['NORMAL', 'PNEUMONIA'],
            'test': ['NORMAL', 'PNEUMONIA'],
            'val': ['NORMAL', 'PNEUMONIA']  # Some versions have validation
        },
        'class_names': ['NORMAL', 'PNEUMONIA'],
        'image_extensions': ['.jpeg', '.jpg', '.png']
    }
    
    # COVID-19 Chest X-Ray Dataset
    COVID_XRAY = {
        'name': 'covid_chest_xray',
        'url': 'https://www.kaggle.com/tawsifurrahman/covid19-radiography-database',
        'expected_structure': {
            'COVID': 'COVID images',
            'Lung_Opacity': 'Lung opacity images', 
            'Normal': 'Normal images',
            'Viral Pneumonia': 'Viral pneumonia images'
        },
        'class_names': ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'],
        'image_extensions': ['.png', '.jpg', '.jpeg']
    }
    
    # Brain MRI Dataset
    BRAIN_MRI = {
        'name': 'brain_mri',
        'url': 'https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri',
        'expected_structure': {
            'Training': ['glioma', 'meningioma', 'notumor', 'pituitary'],
            'Testing': ['glioma', 'meningioma', 'notumor', 'pituitary']
        },
        'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
        'image_extensions': ['.jpg', '.jpeg', '.png']
    }
    
    # NIH Chest X-Rays
    NIH_CHEST_XRAY = {
        'name': 'nih_chest_xray',
        'url': 'https://www.kaggle.com/nih-chest-xrays/data',
        'expected_structure': {
            'images': 'All images in one folder',
            'Data_Entry_2017.csv': 'Labels file'
        },
        'class_names': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                       'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'],
        'image_extensions': ['.png']
    }

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    datasets = {
        'chest_xray': DatasetConfig.CHEST_XRAY_KAGGLE,
        'covid_xray': DatasetConfig.COVID_XRAY,
        'brain_mri': DatasetConfig.BRAIN_MRI,
        'nih_chest_xray': DatasetConfig.NIH_CHEST_XRAY
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name]