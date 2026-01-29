"""
PHASE 2: Model Training
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_model, unfreeze_model
from utils import load_config, set_seed


def load_data(data_path, split_name):
    """Load preprocessed data"""
    file_path = os.path.join(data_path, f"{split_name}.npz")
    data = np.load(file_path)
    return data['X'], data['y']


def create_data_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        layers.RandomRotation(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
    ])


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 50)
    
    # Load config
    config = load_config()
    set_seed(config['seed'])
    
    # Load data
    print("\n1. Loading preprocessed data...")
    data_path = config['data']['processed_path']
    X_train, y_train = load_data(data_path, 'train')
    X_val, y_val = load_data(data_path, 'val')
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    
    # Convert labels to categorical
    num_classes = config['model']['num_classes']
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    
    # Create model
    print("\n2. Creating MobileNetV2 model...")
    model, base_model = create_model(config)
    model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    os.makedirs(os.path.dirname(config['paths']['model_checkpoint']), exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            config['paths']['model_checkpoint'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Train (Phase 1: Frozen base)
    print("\n3. Training (frozen base)...")
    history1 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tune (Phase 2: Unfreeze top layers)
    print("\n4. Fine-tuning (unfrozen top layers)...")
    unfreeze_model(base_model)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['fine_tune_lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_epochs = config['training']['epochs'] + config['training']['fine_tune_epochs']
    history2 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        initial_epoch=len(history1.history['loss']),
        epochs=total_epochs,
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n5. Saving final model...")
    os.makedirs(os.path.dirname(config['paths']['final_model']), exist_ok=True)
    model.save(config['paths']['final_model'])
    print(f"   Saved → {config['paths']['final_model']}")
    
    # Evaluate
    print("\n6. Final evaluation...")
    X_test, y_test = load_data(data_path, 'test')
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print("\n" + "=" * 50)
    print("✓ TRAINING COMPLETE")
    print("=" * 50)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    from tensorflow.keras import layers  # Import for augmentation
    main()