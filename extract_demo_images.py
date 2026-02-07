"""
Extract sample test images from Sign Language MNIST
Creates PNG files for demo
"""
import os
import cv2
import numpy as np
import pandas as pd

# Letters
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def label_to_letter(label):
    """Convert label to letter"""
    if label <= 8:
        return LETTERS[label]
    else:
        return LETTERS[label - 1]

def main():
    print("Extracting test images for demo...")
    
    # Load test data
    test_csv = "data/raw/sign_mnist_test.csv"
    if not os.path.exists(test_csv):
        print(f"❌ {test_csv} not found!")
        return
    
    df = pd.read_csv(test_csv)
    
    # Create output directory
    output_dir = "demo_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract 5 random samples per letter
    samples_per_letter = 3
    
    for letter in LETTERS:
        # Get label
        if letter <= 'I':
            label = ord(letter) - ord('A')
        else:
            label = ord(letter) - ord('A') + 1
        
        # Get samples
        letter_samples = df[df['label'] == label]
        
        if len(letter_samples) == 0:
            continue
        
        # Random sample
        samples = letter_samples.sample(min(samples_per_letter, len(letter_samples)))
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            # Get pixels
            pixels = row.drop('label').values
            
            # Reshape to 28x28
            img = pixels.reshape(28, 28).astype(np.uint8)
            
            # Save
            filename = f"{output_dir}/{letter}_{idx+1}.png"
            cv2.imwrite(filename, img)
            print(f"✓ Saved: {filename}")
    
    print(f"\n✓ Done! {len(os.listdir(output_dir))} images saved to {output_dir}/")
    print("\nYou can now use these images for demo!")

if __name__ == "__main__":
    main()