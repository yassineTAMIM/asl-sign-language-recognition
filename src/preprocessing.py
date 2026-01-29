"""
PHASE 1: Data Preprocessing
- Load CSV data
- Convert to images
- Split train/val/test
- Apply augmentation
"""
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_config, set_seed, label_to_index


def load_csv_data(csv_path):
    """Load Sign Language MNIST CSV"""
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    return pixels, labels


def pixels_to_image(pixel_array, img_size=28):
    """Convert pixel array to image"""
    img = pixel_array.reshape(img_size, img_size)
    return img.astype(np.uint8)


def preprocess_image(img, target_size=64):
    """Resize and normalize image"""
    # Resize
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Convert to RGB (duplicate grayscale)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    img_normalized = img_rgb / 255.0
    
    return img_normalized


def save_processed_data(X, y, split_name, output_dir, config):
    """Save preprocessed data as .npz"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.npz")
    np.savez(output_path, X=X, y=y)
    print(f"✓ Saved {split_name}: {X.shape[0]} samples → {output_path}")


def main():
    """Main preprocessing pipeline"""
    print("=" * 50)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 50)
    
    # Load config
    config = load_config()
    set_seed(config['seed'])
    
    # Paths
    train_csv = os.path.join(config['data']['raw_path'], 'sign_mnist_train.csv')
    test_csv = os.path.join(config['data']['raw_path'], 'sign_mnist_test.csv')
    output_dir = config['data']['processed_path']
    
    # Load training data
    print("\n1. Loading training data...")
    X_train_raw, y_train_raw = load_csv_data(train_csv)
    print(f"   Loaded: {X_train_raw.shape[0]} training samples")
    
    # Load test data
    print("\n2. Loading test data...")
    X_test_raw, y_test_raw = load_csv_data(test_csv)
    print(f"   Loaded: {X_test_raw.shape[0]} test samples")
    
    # Process training images
    print("\n3. Processing images...")
    target_size = config['data']['img_size']
    
    X_train = np.array([
        preprocess_image(pixels_to_image(pixels), target_size)
        for pixels in X_train_raw
    ])
    y_train = np.array([label_to_index(label) for label in y_train_raw])
    
    X_test = np.array([
        preprocess_image(pixels_to_image(pixels), target_size)
        for pixels in X_test_raw
    ])
    y_test = np.array([label_to_index(label) for label in y_test_raw])
    
    print(f"   Processed shape: {X_train.shape}")
    
    # Split train into train/val
    print("\n4. Splitting train/validation...")
    val_ratio = config['data']['val_split'] / (config['data']['train_split'] + config['data']['val_split'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=config['seed'],
        stratify=y_train
    )
    
    # Save processed data
    print("\n5. Saving processed data...")
    save_processed_data(X_train, y_train, "train", output_dir, config)
    save_processed_data(X_val, y_val, "val", output_dir, config)
    save_processed_data(X_test, y_test, "test", output_dir, config)
    
    print("\n" + "=" * 50)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")


if __name__ == "__main__":
    main()