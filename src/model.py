"""
PHASE 2: Model Architecture
MobileNetV2 with Transfer Learning
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# NumPy 2.0 compatibility
np.set_printoptions(legacy='1.25')


def create_model(config):
    """Create MobileNetV2 model with transfer learning"""
    
    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    dropout = config['model']['dropout']
    dense_units = config['model']['dense_units']
    
    # Load pre-trained MobileNetV2 (without top)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout * 0.7)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    return model, base_model


def unfreeze_model(base_model, num_layers=30):
    """Unfreeze top layers for fine-tuning"""
    base_model.trainable = True
    
    # Freeze all except last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    print(f"✓ Unfrozen last {num_layers} layers for fine-tuning")