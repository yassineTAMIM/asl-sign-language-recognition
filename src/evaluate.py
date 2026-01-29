"""
Model Evaluation Script
Generate confusion matrix, classification report, and performance metrics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_config, index_to_letter


def load_data(data_path, split_name):
    """Load preprocessed data"""
    file_path = os.path.join(data_path, f"{split_name}.npz")
    data = np.load(file_path)
    return data['X'], data['y']


def plot_confusion_matrix(cm, classes, filename='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - ASL Recognition', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Confusion matrix saved to {filename}")


def plot_per_class_accuracy(report_dict, classes, filename='per_class_accuracy.png'):
    """Plot per-class accuracy"""
    # Extract f1-scores for each class
    f1_scores = [report_dict[str(i)]['f1-score'] for i in range(len(classes))]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes, f1_scores, color='steelblue', edgecolor='black')
    
    # Color code bars (red for <0.9, yellow for 0.9-0.95, green for >0.95)
    for i, bar in enumerate(bars):
        if f1_scores[i] < 0.90:
            bar.set_color('salmon')
        elif f1_scores[i] < 0.95:
            bar.set_color('gold')
        else:
            bar.set_color('lightgreen')
    
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
    
    plt.title('Per-Class F1-Score', fontsize=16, fontweight='bold')
    plt.xlabel('Class (Letter)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Per-class accuracy saved to {filename}")


def analyze_errors(y_true, y_pred, classes, top_n=10):
    """Analyze most common prediction errors"""
    errors = []
    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            errors.append((classes[true_idx], classes[pred_idx]))
    
    from collections import Counter
    error_counts = Counter(errors)
    
    print("\n" + "=" * 60)
    print(f"TOP {top_n} MOST COMMON ERRORS")
    print("=" * 60)
    print(f"{'True':<6} {'Pred':<6} {'Count':<8} {'Description'}")
    print("-" * 60)
    
    for (true_letter, pred_letter), count in error_counts.most_common(top_n):
        print(f"{true_letter:<6} {pred_letter:<6} {count:<8} {true_letter} confused with {pred_letter}")
    print("=" * 60)


def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load config
    config = load_config()
    classes = config['classes']
    
    # Check if model exists
    model_path = config['paths']['final_model']
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first: python src/train.py"
        )
    
    # Load model
    print("\n1. Loading Model")
    print("-" * 60)
    model = tf.keras.models.load_model(model_path)
    print(f"   ✓ Model loaded from {model_path}")
    
    # Load test data
    print("\n2. Loading Test Data")
    print("-" * 60)
    data_path = config['data']['processed_path']
    X_test, y_test = load_data(data_path, 'test')
    print(f"   ✓ Test data loaded: {X_test.shape}")
    
    # Make predictions
    print("\n3. Making Predictions")
    print("-" * 60)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(f"   ✓ Predictions complete")
    
    # Overall metrics
    print("\n4. Overall Metrics")
    print("-" * 60)
    y_test_cat = tf.keras.utils.to_categorical(y_test, len(classes))
    results = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"   Test Accuracy:      {results[1] * 100:.2f}%")
    print(f"   Test Top-3 Accuracy: {results[2] * 100:.2f}%")
    print(f"   Test Loss:          {results[0]:.4f}")
    
    # Confusion Matrix
    print("\n5. Generating Confusion Matrix")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes)
    
    # Classification Report
    print("\n6. Classification Report")
    print("-" * 60)
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=classes,
        zero_division=0
    ))
    
    # Per-class accuracy plot
    print("\n7. Per-Class Performance")
    print("-" * 60)
    plot_per_class_accuracy(report, classes)
    
    # Error analysis
    print("\n8. Error Analysis")
    print("-" * 60)
    analyze_errors(y_test, y_pred, classes)
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Overall Accuracy: {results[1] * 100:.2f}%")
    print(f"\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - per_class_accuracy.png")
    print("=" * 60)


if __name__ == "__main__":
    main()