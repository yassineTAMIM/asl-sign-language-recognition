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
    
    print(f"   Input shape: {input_shape}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Dropout rate: {dropout}")
    print(f"   Dense units: {dense_units}")
    
    # Load pre-trained MobileNetV2 (without top classification layer)
    print("\n   Loading MobileNetV2 (ImageNet weights)...")
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    print(f"   ✓ Loaded MobileNetV2 ({len(base_model.layers)} layers)")
    print("   ✓ Base model frozen for initial training")
    
    # Build model
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Dropout(dropout, name='dropout_1')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='batch_norm_2')(x)
    x = layers.Dropout(dropout * 0.7, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs, outputs, name='ASL_MobileNetV2')
    
    return model, base_model


def unfreeze_model(base_model, num_layers=30):
    """Unfreeze top layers for fine-tuning"""
    base_model.trainable = True
    
    # Freeze all except last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    
    print(f"\n   ✓ Unfrozen last {num_layers} layers for fine-tuning")
    print(f"   ✓ Trainable layers: {trainable_count}/{len(base_model.layers)}")


def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print("\n" + "=" * 60)
    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 60)