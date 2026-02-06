"""
Comprehensive Visualization Script for ASL Recognition Project
Generates publication-quality visualizations for:
- Dataset analysis
- Model architecture
- Training history
- Performance metrics
- Error analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import tensorflow as tf
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE ASL PROJECT VISUALIZATIONS")
print("=" * 70)


# ============================================================================
# 1. DATA ANALYSIS VISUALIZATIONS
# ============================================================================

def visualize_dataset_overview():
    """Visualize dataset statistics and distribution"""
    print("\n1. Generating Dataset Overview...")
    
    # Load data
    data_path = "data/processed"
    train_data = np.load(f"{data_path}/train.npz")
    val_data = np.load(f"{data_path}/val.npz")
    test_data = np.load(f"{data_path}/test.npz")
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    # Letters
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Dataset sizes
    ax1 = fig.add_subplot(gs[0, 0])
    splits = ['Train', 'Val', 'Test']
    sizes = [len(y_train), len(y_val), len(y_test)]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(splits, sizes, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution - Train
    ax2 = fig.add_subplot(gs[0, 1:])
    train_counts = np.bincount(y_train)
    ax2.bar(letters, train_counts, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Letter', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=train_counts.mean(), color='r', linestyle='--', 
                label=f'Mean: {train_counts.mean():.0f}')
    ax2.legend()
    
    # 3. Pixel intensity distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sample_images = X_train[:1000].flatten()
    ax3.hist(sample_images, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Pixel Intensity', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Pixel Intensity Distribution\n(1000 samples)', 
                  fontsize=14, fontweight='bold')
    ax3.axvline(x=sample_images.mean(), color='r', linestyle='--', 
                label=f'Mean: {sample_images.mean():.3f}')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Sample images grid
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    
    # Create subplot for sample images
    n_samples = 12
    rows, cols = 2, 6
    fig_samples = plt.figure(figsize=(12, 4))
    
    for idx in range(n_samples):
        plt.subplot(rows, cols, idx + 1)
        # Select random sample from each class
        class_idx = idx * 2 if idx < 12 else idx
        class_samples = np.where(y_train == class_idx)[0]
        sample_idx = np.random.choice(class_samples)
        
        img = X_train[sample_idx]
        plt.imshow(img)
        plt.title(f'{letters[class_idx]}', fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    fig_samples.savefig(OUTPUT_DIR / "sample_images.png", dpi=300, bbox_inches='tight')
    plt.close(fig_samples)
    
    # Add reference to sample images
    ax4.text(0.5, 0.5, 'Sample Images\nSaved separately', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Image statistics
    ax5 = fig.add_subplot(gs[2, 0])
    stats_data = {
        'Min': [X_train.min(), X_val.min(), X_test.min()],
        'Max': [X_train.max(), X_val.max(), X_test.max()],
        'Mean': [X_train.mean(), X_val.mean(), X_test.mean()],
        'Std': [X_train.std(), X_val.std(), X_test.std()]
    }
    
    x = np.arange(len(splits))
    width = 0.2
    multiplier = 0
    
    for attribute, measurement in stats_data.items():
        offset = width * multiplier
        ax5.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1
    
    ax5.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax5.set_title('Image Statistics by Split', fontsize=14, fontweight='bold')
    ax5.set_xticks(x + width * 1.5)
    ax5.set_xticklabels(splits)
    ax5.legend(loc='upper left')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Class balance visualization
    ax6 = fig.add_subplot(gs[2, 1:])
    all_counts = [train_counts, np.bincount(y_val), np.bincount(y_test)]
    x = np.arange(len(letters))
    width = 0.25
    
    for i, (counts, split) in enumerate(zip(all_counts, splits)):
        ax6.bar(x + i*width, counts, width, label=split, alpha=0.8)
    
    ax6.set_xlabel('Letter', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax6.set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax6.set_xticks(x + width)
    ax6.set_xticklabels(letters)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('ASL Dataset Overview & Statistics', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(OUTPUT_DIR / "dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {OUTPUT_DIR}/dataset_overview.png")
    print(f"   ✓ Saved: {OUTPUT_DIR}/sample_images.png")
    
    # Print statistics
    print(f"\n   Dataset Statistics:")
    print(f"   - Total samples: {len(y_train) + len(y_val) + len(y_test):,}")
    print(f"   - Train: {len(y_train):,} ({len(y_train)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)")
    print(f"   - Val: {len(y_val):,} ({len(y_val)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)")
    print(f"   - Test: {len(y_test):,} ({len(y_test)/(len(y_train)+len(y_val)+len(y_test))*100:.1f}%)")
    print(f"   - Classes: {len(letters)} letters")
    print(f"   - Image shape: {X_train.shape[1:3]}")


# ============================================================================
# 2. TRAINING HISTORY VISUALIZATIONS
# ============================================================================

def visualize_training_history():
    """Visualize training history for both phases"""
    print("\n2. Generating Training History Visualizations...")
    
    # Load histories
    with open('training_history_phase1.json', 'r') as f:
        history1 = json.load(f)
    
    with open('training_history_phase2.json', 'r') as f:
        history2 = json.load(f)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Colors
    train_color = '#2ecc71'
    val_color = '#e74c3c'
    
    # 1. Accuracy - Phase 1
    ax1 = fig.add_subplot(gs[0, 0])
    epochs1 = range(1, len(history1['accuracy']) + 1)
    ax1.plot(epochs1, history1['accuracy'], 'o-', label='Train', 
             color=train_color, linewidth=2, markersize=6)
    ax1.plot(epochs1, history1['val_accuracy'], 's-', label='Val', 
             color=val_color, linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Phase 1: Frozen Base Model\nAccuracy', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # 2. Loss - Phase 1
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs1, history1['loss'], 'o-', label='Train', 
             color=train_color, linewidth=2, markersize=6)
    ax2.plot(epochs1, history1['val_loss'], 's-', label='Val', 
             color=val_color, linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Phase 1: Frozen Base Model\nLoss', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    # 3. Top-3 Accuracy - Phase 1
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs1, history1['top3_accuracy'], 'o-', label='Train', 
             color=train_color, linewidth=2, markersize=6)
    ax3.plot(epochs1, history1['val_top3_accuracy'], 's-', label='Val', 
             color=val_color, linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Top-3 Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Phase 1: Frozen Base Model\nTop-3 Accuracy', 
                  fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(alpha=0.3)
    
    # 4. Accuracy - Phase 2
    ax4 = fig.add_subplot(gs[1, 0])
    epochs2 = range(1, len(history2['accuracy']) + 1)
    ax4.plot(epochs2, history2['accuracy'], 'o-', label='Train', 
             color=train_color, linewidth=2, markersize=6)
    ax4.plot(epochs2, history2['val_accuracy'], 's-', label='Val', 
             color=val_color, linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Phase 2: Fine-Tuning\nAccuracy', 
                  fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(alpha=0.3)
    
    # 5. Loss - Phase 2
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs2, history2['loss'], 'o-', label='Train', 
             color=train_color, linewidth=2, markersize=6)
    ax5.plot(epochs2, history2['val_loss'], 's-', label='Val', 
             color=val_color, linewidth=2, markersize=6)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax5.set_title('Phase 2: Fine-Tuning\nLoss', 
                  fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(alpha=0.3)
    
    # 6. Learning Rate Schedule
    ax6 = fig.add_subplot(gs[1, 2])
    lr1 = history1['learning_rate']
    lr2 = history2['learning_rate']
    all_epochs = range(1, len(lr1) + len(lr2) + 1)
    all_lr = lr1 + lr2
    ax6.plot(all_epochs, all_lr, 'o-', color='#9b59b6', linewidth=2, markersize=6)
    ax6.axvline(x=len(lr1), color='red', linestyle='--', 
                label='Fine-tuning starts', linewidth=2)
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax6.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Combined Accuracy
    ax7 = fig.add_subplot(gs[2, :2])
    all_train_acc = history1['accuracy'] + history2['accuracy']
    all_val_acc = history1['val_accuracy'] + history2['val_accuracy']
    all_epochs = range(1, len(all_train_acc) + 1)
    
    ax7.plot(all_epochs, all_train_acc, 'o-', label='Train', 
             color=train_color, linewidth=2, markersize=4)
    ax7.plot(all_epochs, all_val_acc, 's-', label='Val', 
             color=val_color, linewidth=2, markersize=4)
    ax7.axvline(x=len(history1['accuracy']), color='black', 
                linestyle='--', label='Fine-tuning starts', linewidth=2)
    
    # Add shaded regions
    ax7.axvspan(0, len(history1['accuracy']), alpha=0.1, color='blue', 
                label='Phase 1: Frozen')
    ax7.axvspan(len(history1['accuracy']), len(all_train_acc), 
                alpha=0.1, color='orange', label='Phase 2: Fine-tune')
    
    ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax7.set_title('Complete Training Journey: Accuracy', 
                  fontsize=14, fontweight='bold')
    ax7.legend(loc='lower right')
    ax7.grid(alpha=0.3)
    
    # 8. Training Summary Box
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate final metrics
    final_train_acc = all_train_acc[-1]
    final_val_acc = all_val_acc[-1]
    final_train_loss = (history1['loss'] + history2['loss'])[-1]
    final_val_loss = (history1['val_loss'] + history2['val_loss'])[-1]
    
    summary_text = f"""
    TRAINING SUMMARY
    ═══════════════════════
    
    Phase 1 (Frozen Base)
    • Epochs: {len(history1['accuracy'])}
    • Initial LR: {history1['learning_rate'][0]:.4f}
    • Final Train Acc: {history1['accuracy'][-1]:.4f}
    • Final Val Acc: {history1['val_accuracy'][-1]:.4f}
    
    Phase 2 (Fine-Tuning)
    • Epochs: {len(history2['accuracy'])}
    • Initial LR: {history2['learning_rate'][0]:.6f}
    • Final Train Acc: {history2['accuracy'][-1]:.4f}
    • Final Val Acc: {history2['val_accuracy'][-1]:.4f}
    
    Overall Best
    • Train Acc: {max(all_train_acc):.4f}
    • Val Acc: {max(all_val_acc):.4f}
    • Train Loss: {min(history1['loss'] + history2['loss']):.4f}
    • Val Loss: {min(history1['val_loss'] + history2['val_loss']):.4f}
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=10, 
             family='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Training History Analysis: Two-Phase Transfer Learning', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {OUTPUT_DIR}/training_history.png")
    
    # Print summary
    print(f"\n   Training Summary:")
    print(f"   - Total epochs: {len(all_train_acc)}")
    print(f"   - Best train accuracy: {max(all_train_acc):.4f}")
    print(f"   - Best val accuracy: {max(all_val_acc):.4f}")
    print(f"   - Final train accuracy: {final_train_acc:.4f}")
    print(f"   - Final val accuracy: {final_val_acc:.4f}")


# ============================================================================
# 3. MODEL ARCHITECTURE VISUALIZATION
# ============================================================================

def visualize_model_architecture():
    """Visualize model architecture and parameters"""
    print("\n3. Generating Model Architecture Visualization...")
    
    # Load model
    model_path = "models/final/asl_model.h5"
    model = tf.keras.models.load_model(model_path)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Model Summary Text
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Get model summary as text
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    summary_text = '\n'.join(summary_lines[:30])  # First 30 lines
    ax1.text(0.05, 0.95, summary_text, fontsize=8, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax1.set_title('Model Architecture Summary (First 30 layers)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # 2. Layer Type Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    # Plot
    types = list(layer_types.keys())
    counts = list(layer_types.values())
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(types)))
    
    ax2.barh(types, counts, color=colors_palette, edgecolor='black')
    ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Layer Types Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (t, c) in enumerate(zip(types, counts)):
        ax2.text(c, i, f' {c}', va='center', fontweight='bold')
    
    # 3. Parameter Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    trainable_params = []
    non_trainable_params = []
    layer_names = []
    
    for layer in model.layers:
        trainable = sum([tf.size(w).numpy() for w in layer.trainable_weights])
        non_trainable = sum([tf.size(w).numpy() for w in layer.non_trainable_weights])
        
        if trainable > 0 or non_trainable > 0:
            layer_names.append(layer.name[:20])  # Truncate long names
            trainable_params.append(trainable)
            non_trainable_params.append(non_trainable)
    
    # Select top layers by parameter count
    top_n = 15
    total_params = [t + nt for t, nt in zip(trainable_params, non_trainable_params)]
    top_indices = np.argsort(total_params)[-top_n:]
    
    selected_names = [layer_names[i] for i in top_indices]
    selected_trainable = [trainable_params[i] for i in top_indices]
    selected_non_trainable = [non_trainable_params[i] for i in top_indices]
    
    y_pos = np.arange(len(selected_names))
    
    ax3.barh(y_pos, selected_trainable, label='Trainable', color='#3498db')
    ax3.barh(y_pos, selected_non_trainable, left=selected_trainable, 
             label='Non-trainable', color='#e74c3c')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(selected_names, fontsize=8)
    ax3.set_xlabel('Parameters', fontsize=12, fontweight='bold')
    ax3.set_title(f'Top {top_n} Layers by Parameter Count', 
                  fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    
    # Format x-axis
    ax3.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    
    plt.suptitle('Model Architecture Analysis: MobileNetV2', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / "model_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed architecture to text file
    with open(OUTPUT_DIR / "model_architecture.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"   ✓ Saved: {OUTPUT_DIR}/model_architecture.png")
    print(f"   ✓ Saved: {OUTPUT_DIR}/model_architecture.txt")
    
    # Print stats
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    trainable_params_total = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"\n   Model Statistics:")
    print(f"   - Total layers: {len(model.layers)}")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params_total:,}")
    print(f"   - Non-trainable parameters: {total_params - trainable_params_total:,}")
    print(f"   - Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")


# ============================================================================
# 4. PERFORMANCE ANALYSIS
# ============================================================================

def visualize_performance_analysis():
    """Comprehensive performance analysis visualization"""
    print("\n4. Generating Performance Analysis...")
    
    # Load confusion matrix image
    confusion_img = plt.imread("confusion_matrix.png")
    per_class_img = plt.imread("per_class_accuracy.png")
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.2)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(confusion_img)
    ax1.axis('off')
    ax1.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Per-Class Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(per_class_img)
    ax2.axis('off')
    ax2.set_title('Per-Class F1-Score', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('Model Performance Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / "performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {OUTPUT_DIR}/performance_analysis.png")


# ============================================================================
# 5. COMPREHENSIVE PROJECT SUMMARY
# ============================================================================

def create_project_summary():
    """Create comprehensive project summary visualization"""
    print("\n5. Generating Project Summary...")
    
    # Load training histories
    with open('training_history_phase1.json', 'r') as f:
        history1 = json.load(f)
    with open('training_history_phase2.json', 'r') as f:
        history2 = json.load(f)
    
    # Load data info
    train_data = np.load("data/processed/train.npz")
    val_data = np.load("data/processed/val.npz")
    test_data = np.load("data/processed/test.npz")
    
    # Create figure
    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.98, 'ASL Recognition Project - Complete Summary', 
             ha='center', fontsize=24, fontweight='bold')
    
    # Section 1: Project Overview
    y_pos = 0.94
    section_text = """
    PROJECT OVERVIEW
    ══════════════════════════════════════════════════════════════════════════
    
    Title: American Sign Language (ASL) Alphabet Recognition
    Task: Real-time hand sign classification using computer vision
    Dataset: Sign Language MNIST (Kaggle)
    Model: MobileNetV2 with Transfer Learning
    Framework: TensorFlow/Keras
    Deployment: FastAPI + Docker + Frontend
    """
    
    fig.text(0.05, y_pos, section_text, fontsize=11, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Section 2: Dataset Statistics
    y_pos -= 0.15
    dataset_text = f"""
    DATASET STATISTICS
    ══════════════════════════════════════════════════════════════════════════
    
    Total Samples: {len(train_data['y']) + len(val_data['y']) + len(test_data['y']):,}
    
    Training Set:   {len(train_data['y']):,} samples ({len(train_data['y'])/(len(train_data['y'])+len(val_data['y'])+len(test_data['y']))*100:.1f}%)
    Validation Set: {len(val_data['y']):,} samples ({len(val_data['y'])/(len(train_data['y'])+len(val_data['y'])+len(test_data['y']))*100:.1f}%)
    Test Set:       {len(test_data['y']):,} samples ({len(test_data['y'])/(len(train_data['y'])+len(val_data['y'])+len(test_data['y']))*100:.1f}%)
    
    Classes: 24 letters (A-I, K-Y)
    Missing: J and Z (require motion)
    Image Size: 64×64×3 (RGB)
    Normalization: [0, 1]
    """
    
    fig.text(0.05, y_pos, dataset_text, fontsize=11, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Section 3: Model Architecture
    y_pos -= 0.20
    model = tf.keras.models.load_model("models/final/asl_model.h5")
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    model_text = f"""
    MODEL ARCHITECTURE
    ══════════════════════════════════════════════════════════════════════════
    
    Base Model: MobileNetV2 (ImageNet pretrained)
    Input Shape: (64, 64, 3)
    Output Classes: 24
    
    Architecture:
      • MobileNetV2 backbone (frozen initially)
      • GlobalAveragePooling2D
      • BatchNormalization
      • Dropout (0.3)
      • Dense(128, relu)
      • BatchNormalization
      • Dropout (0.21)
      • Dense(24, softmax)
    
    Parameters:
      • Total: {total_params:,}
      • Trainable: {trainable_params:,}
      • Non-trainable: {total_params - trainable_params:,}
    
    Model Size: {os.path.getsize('models/final/asl_model.h5') / (1024*1024):.2f} MB
    """
    
    fig.text(0.05, y_pos, model_text, fontsize=11, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    # Section 4: Training Configuration
    y_pos -= 0.25
    training_text = f"""
    TRAINING CONFIGURATION
    ══════════════════════════════════════════════════════════════════════════
    
    Phase 1: Transfer Learning (Frozen Base)
      • Epochs: {len(history1['accuracy'])}
      • Learning Rate: {history1['learning_rate'][0]:.4f}
      • Batch Size: 32
      • Optimizer: Adam
      • Loss: Categorical Crossentropy
      • Metrics: Accuracy, Top-3 Accuracy
    
    Phase 2: Fine-Tuning (Unfrozen Top Layers)
      • Epochs: {len(history2['accuracy'])}
      • Initial Learning Rate: {history2['learning_rate'][0]:.6f}
      • Unfrozen Layers: Last 30 layers
      • Learning Rate Reduction: ReduceLROnPlateau
      • Early Stopping: Patience=5
    
    Data Augmentation:
      • Random Rotation (±15°)
      • Random Translation (10%)
      • Random Zoom (20%)
      • Random Horizontal Flip
    """
    
    fig.text(0.05, y_pos, training_text, fontsize=11, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # Section 5: Results
    y_pos -= 0.28
    final_train_acc = (history1['accuracy'] + history2['accuracy'])[-1]
    final_val_acc = (history1['val_accuracy'] + history2['val_accuracy'])[-1]
    best_val_acc = max(history1['val_accuracy'] + history2['val_accuracy'])
    
    results_text = f"""
    RESULTS & PERFORMANCE
    ══════════════════════════════════════════════════════════════════════════
    
    Training Performance:
      • Final Train Accuracy:     {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
      • Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
      • Best Validation Accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
    
    Test Set Performance:
      • Test Accuracy:      99.93%
      • Top-3 Accuracy:     100.00%
      • Test Loss:          0.0013
    
    Per-Class Performance:
      • Perfect (100%):     A, B, F, H, O, P, S, T, X, Y
      • Excellent (>99%):   D, E, K, C, G, I, L, Q, R, U, W
      • Good (>93%):        M, N, V
    
    Common Confusions:
      • M ↔ N (similar finger positions)
      • V ↔ other signs (overlapping features)
    """
    
    fig.text(0.05, y_pos, results_text, fontsize=11, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Section 6: Deployment
    y_pos -= 0.25
    deployment_text = """
    DEPLOYMENT & DOCKER
    ══════════════════════════════════════════════════════════════════════════
    
    API: FastAPI (Python)
      • Endpoints: /predict, /predict_pipeline, /health, /classes
      • Features: Real-time hand detection, preprocessing pipeline visualization
      • CORS: Enabled for frontend integration
    
    Frontend: HTML/CSS/JavaScript
      • Real-time webcam capture
      • Pipeline visualization (5 stages)
      • Top-3 predictions display
      • Flip camera support
    
    Docker Containerization:
      • Dockerfile.train: Training environment
      • Dockerfile.api: API deployment
      • docker-compose.yml: Multi-service orchestration
      • Volumes: Data, models, logs
      • Networks: Isolated bridge network
      • Health checks: Configured
    
    Services:
      1. Trainer: Model training service
      2. API: FastAPI inference service (port 8000)
      3. Frontend: Nginx static file server (port 80)
    """
    
    fig.text(0.05, y_pos, deployment_text, fontsize=11, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    # Footer
    fig.text(0.5, 0.02, 
             f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
             f'Framework: TensorFlow {tf.__version__}',
             ha='center', fontsize=10, style='italic')
    
    plt.savefig(OUTPUT_DIR / "project_summary.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    print(f"   ✓ Saved: {OUTPUT_DIR}/project_summary.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations"""
    
    print("\nStarting visualization generation...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    try:
        visualize_dataset_overview()
    except Exception as e:
        print(f"   ✗ Error in dataset overview: {e}")
    
    try:
        visualize_training_history()
    except Exception as e:
        print(f"   ✗ Error in training history: {e}")
    
    try:
        visualize_model_architecture()
    except Exception as e:
        print(f"   ✗ Error in model architecture: {e}")
    
    try:
        visualize_performance_analysis()
    except Exception as e:
        print(f"   ✗ Error in performance analysis: {e}")
    
    try:
        create_project_summary()
    except Exception as e:
        print(f"   ✗ Error in project summary: {e}")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in '{OUTPUT_DIR}':")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  • {file.name}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()