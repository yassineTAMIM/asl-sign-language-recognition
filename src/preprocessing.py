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
from tqdm import tqdm
from utils import load_config, set_seed, label_to_index, validate_data_structure, print_data_info


def load_csv_data(csv_path):
    """Load Sign Language MNIST CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    print(f"   Loading from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate CSV structure
    assert 'label' in df.columns, "CSV missing 'label' column"
    assert df.shape[1] == 785, "Expected 785 columns (1 label + 784 pixels)"
    
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values
    
    # Validate data
    assert pixels.shape[1] == 784, "Expected 784 pixels (28x28)"
    assert labels.min() >= 0 and labels.max() <= 23, f"Invalid labels range: {labels.min()}-{labels.max()}"
    
    print(f"   ✓ Loaded {len(labels)} samples")
    print(f"   ✓ Unique labels: {np.unique(labels)}")
    
    return pixels, labels


def pixels_to_image(pixel_array, img_size=28):
    """Convert pixel array to image"""
    img = pixel_array.reshape(img_size, img_size)
    return img.astype(np.uint8)


def preprocess_image(img, target_size=64):
    """Resize and normalize image"""
    # Resize
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Convert to RGB (duplicate grayscale to 3 channels)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb / 255.0
    
    return img_normalized


def process_images_batch(pixel_arrays, target_size, desc="Processing"):
    """Process multiple images with progress bar"""
    processed = []
    for pixels in tqdm(pixel_arrays, desc=desc):
        img = pixels_to_image(pixels)
        processed.append(preprocess_image(img, target_size))
    return np.array(processed)


def save_processed_data(X, y, split_name, output_dir, config):
    """Save preprocessed data as .npz"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.npz")
    np.savez_compressed(output_path, X=X, y=y)
    print(f"   ✓ Saved {split_name}: {X.shape[0]} samples → {output_path}")


def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 60)
    
    # Load config
    config = load_config()
    set_seed(config['seed'])
    validate_data_structure(config)
    
    # Paths
    train_csv = os.path.join(config['data']['raw_path'], 'sign_mnist_train.csv')
    test_csv = os.path.join(config['data']['raw_path'], 'sign_mnist_test.csv')
    output_dir = config['data']['processed_path']
    target_size = config['data']['img_size']
    
    # Load training data
    print("\n1. Loading Training Data")
    print("-" * 60)
    X_train_raw, y_train_raw = load_csv_data(train_csv)
    
    # Load test data
    print("\n2. Loading Test Data")
    print("-" * 60)
    X_test_raw, y_test_raw = load_csv_data(test_csv)
    
    # Process training images
    print("\n3. Processing Images")
    print("-" * 60)
    print(f"   Target size: {target_size}x{target_size}")
    print(f"   Output channels: 3 (RGB)")
    
    X_train = process_images_batch(X_train_raw, target_size, "Processing train")
    y_train = np.array([label_to_index(label) for label in y_train_raw])
    
    X_test = process_images_batch(X_test_raw, target_size, "Processing test")
    y_test = np.array([label_to_index(label) for label in y_test_raw])
    
    print(f"\n   ✓ Processed shape: {X_train.shape}")
    
    # Validate processed data
    print("\n4. Validating Processed Data")
    print("-" * 60)
    assert X_train.min() >= 0 and X_train.max() <= 1, "Invalid normalization"
    assert X_train.shape[1:] == (target_size, target_size, 3), "Invalid shape"
    print("   ✓ Normalization correct [0, 1]")
    print("   ✓ Shape correct")
    print("   ✓ All validations passed")
    
    # Split train into train/val
    print("\n5. Splitting Train/Validation")
    print("-" * 60)
    val_ratio = config['data']['val_split'] / (config['data']['train_split'] + config['data']['val_split'])
    print(f"   Validation ratio: {val_ratio:.2%}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=config['seed'],
        stratify=y_train
    )
    
    # Print dataset statistics
    print_data_info(X_train, y_train, "Train")
    print_data_info(X_val, y_val, "Validation")
    print_data_info(X_test, y_test, "Test")
    
    # Save processed data
    print("\n6. Saving Processed Data")
    print("-" * 60)
    save_processed_data(X_train, y_train, "train", output_dir, config)
    save_processed_data(X_val, y_val, "val", output_dir, config)
    save_processed_data(X_test, y_test, "test", output_dir, config)
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Train: {X_train.shape[0]:,} samples")
    print(f"Val:   {X_val.shape[0]:,} samples")
    print(f"Test:  {X_test.shape[0]:,} samples")
    print(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,} samples")
    print(f"\nData saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()