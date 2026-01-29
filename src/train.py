"""
PHASE 2: Model Training
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_model, unfreeze_model, print_model_summary
from utils import load_config, set_seed


def load_data(data_path, split_name):
    """Load preprocessed data"""
    file_path = os.path.join(data_path, f"{split_name}.npz")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Preprocessed data not found: {file_path}\n"
            f"Please run preprocessing first: python src/preprocessing.py"
        )
    
    data = np.load(file_path)
    return data['X'], data['y']


def create_data_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        layers.RandomRotation(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
    ], name='data_augmentation')


class TrainingLogger(tf.keras.callbacks.Callback):
    """Custom callback to log training progress"""
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.4f}")
        print(f"  Val Loss:   {logs['val_loss']:.4f} | Val Acc:   {logs['val_accuracy']:.4f}")
        print(f"  LR: {float(self.model.optimizer.lr):.6f}")


def save_history(history, filename='training_history.json'):
    """Save training history to JSON"""
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(filename, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"\n   ✓ Training history saved to {filename}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 60)
    
    # Load config
    config = load_config()
    set_seed(config['seed'])
    
    # Load data
    print("\n1. Loading Preprocessed Data")
    print("-" * 60)
    data_path = config['data']['processed_path']
    X_train, y_train = load_data(data_path, 'train')
    X_val, y_val = load_data(data_path, 'val')
    
    print(f"   Train: {X_train.shape} | Labels: {y_train.shape}")
    print(f"   Val:   {X_val.shape} | Labels: {y_val.shape}")
    print(f"   ✓ Data loaded successfully")
    
    # Convert labels to categorical
    num_classes = config['model']['num_classes']
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    print(f"   ✓ Labels converted to categorical (one-hot)")
    
    # Create model
    print("\n2. Creating MobileNetV2 Model")
    print("-" * 60)
    model, base_model = create_model(config)
    print_model_summary(model)
    
    # Compile
    print("\n3. Compiling Model")
    print("-" * 60)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['training']['learning_rate']
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    print(f"   Optimizer: Adam (lr={config['training']['learning_rate']})")
    print(f"   Loss: Categorical Crossentropy")
    print(f"   Metrics: Accuracy, Top-3 Accuracy")
    print("   ✓ Model compiled")
    
    # Callbacks
    print("\n4. Setting Up Callbacks")
    print("-" * 60)
    os.makedirs(os.path.dirname(config['paths']['model_checkpoint']), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            config['paths']['model_checkpoint'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
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
            min_lr=1e-7,
            verbose=1
        ),
        TrainingLogger()
    ]
    print("   ✓ Callbacks configured:")
    print("     - ModelCheckpoint (save best model)")
    print("     - EarlyStopping (patience=5)")
    print("     - ReduceLROnPlateau (factor=0.5, patience=3)")
    print("     - TrainingLogger (custom)")
    
    # Train Phase 1: Frozen base
    print("\n5. Training Phase 1: Frozen Base Model")
    print("-" * 60)
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    
    history1 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=2
    )
    
    # Save Phase 1 history
    save_history(history1, 'training_history_phase1.json')
    
    # Fine-tune Phase 2: Unfreeze top layers
    print("\n6. Training Phase 2: Fine-Tuning")
    print("-" * 60)
    unfreeze_model(base_model)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['training']['fine_tune_lr']
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    print(f"   Fine-tune LR: {config['training']['fine_tune_lr']}")
    
    total_epochs = config['training']['epochs'] + config['training']['fine_tune_epochs']
    
    history2 = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        initial_epoch=len(history1.history['loss']),
        epochs=total_epochs,
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=2
    )
    
    # Save Phase 2 history
    save_history(history2, 'training_history_phase2.json')
    
    # Save final model
    print("\n7. Saving Final Model")
    print("-" * 60)
    os.makedirs(os.path.dirname(config['paths']['final_model']), exist_ok=True)
    model.save(config['paths']['final_model'])
    print(f"   ✓ Model saved to: {config['paths']['final_model']}")
    
    # Model size
    model_size = os.path.getsize(config['paths']['final_model']) / (1024 * 1024)
    print(f"   Model size: {model_size:.2f} MB")
    
    # Final Evaluation
    print("\n8. Final Evaluation on Test Set")
    print("-" * 60)
    X_test, y_test = load_data(data_path, 'test')
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    test_results = model.evaluate(X_test, y_test_cat, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1]
    test_top3 = test_results[2]
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy:      {test_acc * 100:.2f}%")
    print(f"Final Test Top-3 Accuracy: {test_top3 * 100:.2f}%")
    print(f"Final Test Loss:          {test_loss:.4f}")
    print(f"Model Size:               {model_size:.2f} MB")
    print(f"\nModel saved to: {config['paths']['final_model']}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run evaluation: python src/evaluate.py")
    print("  2. Start API: uvicorn api.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()