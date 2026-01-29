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
    
    The Sign Language MNIST dataset has 24 classes (0-23)
    with J(9) and Z(25) already excluded from the original alphabet.
    Labels are: 0=A, 1=B, 2=C, ..., 8=I, 9=K (not J), ..., 23=Y (not Z)
    
    So we use direct mapping without any adjustment.
    """
    return int(label)


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