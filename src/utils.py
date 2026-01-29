"""Utility functions"""
import yaml
import numpy as np
import tensorflow as tf
import random
import os


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed}")


def get_class_mapping(config):
    """Get class index to letter mapping"""
    classes = config['classes']
    return {i: letter for i, letter in enumerate(classes)}


def label_to_index(label):
    """Convert dataset label to class index.
    
    The Sign Language MNIST dataset labels are 0-24 (25 labels total):
    - 0-8: A-I (9 letters)
    - 9: J (MISSING - no data for J)
    - 10-24: K-Y (15 letters)
    - 25: Z (MISSING - no data for Z)
    
    We need to map these 24 actual labels to indices 0-23:
    - 0-8 → 0-8 (A-I)
    - 10-24 → 9-23 (K-Y)
    """
    label = int(label)
    
    # If label is 0-8, keep as is (A-I)
    if label <= 8:
        return label
    # If label is 10-24, shift down by 1 (K-Y become indices 9-23)
    elif label >= 10:
        return label - 1
    else:
        # This should never happen (label 9 = J doesn't exist in dataset)
        raise ValueError(f"Invalid label {label}. Label 9 (J) should not exist in dataset.")


def index_to_letter(index, config):
    """Convert class index to letter"""
    mapping = get_class_mapping(config)
    if index not in mapping:
        raise ValueError(f"Invalid index {index}. Must be 0-23")
    return mapping[index]


def validate_data_structure(config):
    """Validate that required directories exist"""
    paths_to_check = [
        config['data']['raw_path'],
        config['data']['processed_path'],
        os.path.dirname(config['paths']['model_checkpoint']),
        os.path.dirname(config['paths']['final_model'])
    ]
    
    for path in paths_to_check:
        os.makedirs(path, exist_ok=True)
    
    print("✓ Directory structure validated")


def print_data_info(X, y, split_name):
    """Print dataset information"""
    print(f"\n{split_name} Data:")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Min/Max: [{X.min():.3f}, {X.max():.3f}]")