"""Utility functions"""
import yaml
import numpy as np
import tensorflow as tf
import random


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_class_mapping(config):
    """Get class index to letter mapping"""
    classes = config['classes']
    return {i: letter for i, letter in enumerate(classes)}


def label_to_index(label):
    """Convert label (0-25) to class index (0-23)
    Skip J(9) and Z(25)"""
    label = int(label)  # Ensure integer type
    if label >= 9:
        label -= 1
    if label >= 25:
        label -= 1
    return int(label)


def index_to_letter(index, config):
    """Convert class index to letter"""
    mapping = get_class_mapping(config)
    return mapping[index]