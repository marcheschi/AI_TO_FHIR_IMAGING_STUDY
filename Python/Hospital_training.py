#!/usr/bin/env python
# coding: utf-8

"""
Hospital Training Script for Cardiac Pathologies Detection
======================================================

This script trains a Convolutional Neural Network (CNN) to identify aortic calcification 
and other cardiac pathologies from JPEG images. It supports both a custom CNN model 
and transfer learning using various pre-trained architectures (ResNet, VGG, etc.).

Features:
- K-Fold Cross-Validation
- Transfer Learning with Fine-Tuning
- Grad-CAM Visualization
- Comprehensive Metrics Reporting (CSV, Plots, Confusion Matrix)

Usage:
    python Hospital_training_cleaned.py --data_dir /path/to/data --output_dir /path/to/output

"""

import os
import sys
import argparse
import time
import json
import random
import shutil
import csv
import io
import re
import socket
from datetime import datetime
from contextlib import redirect_stdout
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
from PIL import Image, ImageDraw, ImageFont
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, BatchNormalization, Dropout, Conv2D, GlobalAveragePooling2D, MaxPooling2D
)
from tensorflow.keras.applications import (
    VGG16, VGG19, Xception, InceptionV3, MobileNetV2, DenseNet201, 
    NASNetLarge, InceptionResNetV2, ResNet152V2, ResNet50
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve
)

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train CNN for Cardiac Pathologies Detection")
    
    # Default path logic based on hostname (preserved from original script for backward compatibility)
    hostname = socket.gethostname()
    default_data_dir = "."
    if hostname == 'Hospital':
        default_data_dir = 'Z:/Hospital/_ADDESTRAMENTO'
    elif hostname == 'DESKTOP':
        default_data_dir = 'D:/Hospital/Hospital/_ADDESTRAMENTO'
    
    parser.add_argument('--data_dir', type=str, default=default_data_dir, 
                        help='Path to the directory containing image subdirectories')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Path to save results. Defaults to data_dir if not specified.')
    parser.add_argument('--img_width', type=int, default=400, help='Target image width')
    parser.add_argument('--img_height', type=int, default=400, help='Target image height')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for K-Fold CV')
    parser.add_argument('--mode', type=str, choices=['kfold', 'single', 'transfer'], default='kfold',
                        help='Training mode: kfold (Cross-Validation), single (One split), transfer (Transfer Learning)')
    
    return parser.parse_args()

def setup_directories(base_dir):
    """Creates necessary output directories."""
    dirs_to_create = [
        "KFold_results/models",
        "results/images",
        "results/metrics",
        "results/plots",
        "results/gradcam",
        "MODELS"
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    print(f"Created output directories in {base_dir}")

def get_data_info(main_dir):
    """Scans the directory to find classes and count images."""
    subdirs = [name for name in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, name))]
    subdirs_with_jpg = [subdir for subdir in subdirs 
                        if any(fname.lower().endswith(('.jpg', '.jpeg')) 
                               for fname in os.listdir(os.path.join(main_dir, subdir)))]
    
    labels = {subdir: idx for idx, subdir in enumerate(subdirs_with_jpg)}
    
    jpg_file_counts = {}
    for subdir in subdirs_with_jpg:
        subdir_path = os.path.join(main_dir, subdir)
        count = len([f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg'))])
        jpg_file_counts[subdir] = count
        
    return labels, subdirs_with_jpg, jpg_file_counts

def load_and_preprocess_images(main_dir, subdirs, img_width, img_height):
    """Loads images into numpy arrays."""
    data = []
    labels_list = []
    
    print("Loading images...")
    for i, subdir in enumerate(subdirs):
        path = os.path.join(main_dir, subdir)
        img_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for img_file in img_files:
            try:
                image_path = os.path.join(path, img_file)
                image = Image.open(image_path)
                image = image.resize((img_width, img_height))
                image = image.convert('RGB')
                image = np.array(image)
                data.append(image)
                labels_list.append(i)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                
    return np.array(data), np.array(labels_list)


def create_custom_model(img_width, img_height, num_classes, activation='softmax'):
    """Defines the custom CNN architecture used in the original script."""
    model = keras.Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        Dropout(0.1),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(128, kernel_size=(3,3), activation='relu'),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2,2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation=activation)
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transfer_model(base_model_name, img_width, img_height, num_classes, 
                          n_neurons=256, dropout_rate=0.5, activation='softmax'):
    """Creates a model based on a pre-trained architecture."""
    
    base_models_dict = {
        "ResNet152V2": ResNet152V2,
        "ResNet50": ResNet50,
        "DenseNet201": DenseNet201,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "Xception": Xception,
        "InceptionV3": InceptionV3,
        "MobileNetV2": MobileNetV2,
        "NASNetLarge": NASNetLarge,
    }
    
    if base_model_name not in base_models_dict:
        raise ValueError(f"Model {base_model_name} not found. Available: {list(base_models_dict.keys())}")
        
    BaseModel = base_models_dict[base_model_name]
    base_model = BaseModel(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    
    # Freeze base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_neurons, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation=activation)(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
    
    return model, base_model

def find_last_conv_layer(model):
    """Finds the last Conv2D layer in the model for Grad-CAM."""
    # Ensure model is built
    if not model.built:
        try:
            # Try to build with the input shape from the first layer
            if hasattr(model, 'layers') and len(model.layers) > 0:
                input_shape = model.layers[0].input_shape
                # If input_shape is (None, h, w, c), use it. If (h, w, c), add None.
                if len(input_shape) == 3:
                    model.build((None,) + input_shape)
                else:
                    model.build(input_shape)
        except Exception as e:
            print(f"Warning: Could not build model in find_last_conv_layer: {e}")
            pass

    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    return None

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Generates a Grad-CAM heatmap."""
    try:
        # Strategy 1: Standard Model construction
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        # Strategy 2: Functional Reconstruction
        # This creates a new graph using the same layers (shared weights)
        try:
            input_shape = img_array.shape[1:]
            new_input = Input(shape=input_shape)
            x = new_input
            target_layer_output = None
            
            for layer in model.layers:
                x = layer(x)
                if layer.name == last_conv_layer_name:
                    target_layer_output = x
            
            if target_layer_output is None:
                raise ValueError(f"Layer {last_conv_layer_name} not found in model during reconstruction")
                
            grad_model = Model(inputs=new_input, outputs=[target_layer_output, x])
            
        except Exception as e2:
            print(f"Grad-CAM failed: {e2}")
            return np.zeros((img_array.shape[1], img_array.shape[2]))
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        print(f"Warning: Gradients are None for layer {last_conv_layer_name}. Skipping Grad-CAM for this image.")
        return np.zeros((last_conv_layer_output.shape[1], last_conv_layer_output.shape[2]))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_gradcam(model, img_array, original_img, save_path, layer_name):
    """Saves Grad-CAM visualization."""
    try:
        heatmap = get_gradcam_heatmap(model, img_array, layer_name)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(save_path, superimposed_img)
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")

def plot_confusion_matrix(cm, classes, save_path):
    """Plots and saves confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results_df, save_path, title="Model Comparison"):
    """Creates a bar plot comparing different models.
    
    Args:
        results_df: DataFrame with columns 'Model', 'Test Accuracy', 'Test Loss', 'Time'
        save_path: Path to save the comparison plot
        title: Title for the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy comparison
    axes[0].barh(results_df['Model'], results_df['Test Accuracy'], color='skyblue')
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlim([0, 1])
    for i, (model, acc) in enumerate(zip(results_df['Model'], results_df['Test Accuracy'])):
        axes[0].text(acc + 0.01, i, f'{acc:.4f}', va='center')
    
    # Loss comparison
    axes[1].barh(results_df['Model'], results_df['Test Loss'], color='lightcoral')
    axes[1].set_xlabel('Test Loss')
    axes[1].set_title('Model Loss Comparison')
    for i, (model, loss) in enumerate(zip(results_df['Model'], results_df['Test Loss'])):
        axes[1].text(loss + 0.01, i, f'{loss:.4f}', va='center')
    
    # Training time comparison
    axes[2].barh(results_df['Model'], results_df['Time'], color='lightgreen')
    axes[2].set_xlabel('Training Time (seconds)')
    axes[2].set_title('Model Training Time Comparison')
    for i, (model, time_val) in enumerate(zip(results_df['Model'], results_df['Time'])):
        axes[2].text(time_val + 1, i, f'{time_val:.1f}s', va='center')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")

def save_training_plots(history, save_dir, prefix=""):
    """Saves accuracy and loss plots."""
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'{prefix} Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{prefix}accuracy.png'))
    plt.close()
    
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{prefix} Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{prefix}loss.png'))
    plt.close()

def run_kfold_training(args, X, Y, labels, target_names):
    """Executes K-Fold Cross-Validation."""
    print(f"Starting {args.k_folds}-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        print(f"\n--- FOLD {fold+1}/{args.k_folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        # Convert labels to categorical
        Y_train_cat = to_categorical(Y_train, num_classes=len(labels))
        Y_val_cat = to_categorical(Y_val, num_classes=len(labels))
        
        # Data Augmentation
        datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=30, zoom_range=0.2,
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = datagen.flow(X_train, Y_train_cat, batch_size=args.batch_size)
        val_generator = val_datagen.flow(X_val, Y_val_cat, batch_size=args.batch_size, shuffle=False)
        
        # Create and Train Model
        model = create_custom_model(args.img_width, args.img_height, len(labels))
        
        callbacks = [
            EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
            ReduceLROnPlateau(patience=5, monitor='val_loss', factor=0.2, verbose=1)
        ]
        
        start_time = time.time()
        history = model.fit(
            train_generator, 
            epochs=args.epochs, 
            validation_data=val_generator, 
            callbacks=callbacks,
            verbose=1
        )
        elapsed_time = time.time() - start_time
        
        # Evaluation
        val_loss, val_acc = model.evaluate(val_generator)
        print(f"Fold {fold+1} Accuracy: {val_acc:.4f}")
        
        # Save Results
        fold_dir = os.path.join(args.output_dir, 'KFold_results', f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        save_training_plots(history, fold_dir)
        
        # Confusion Matrix
        Y_pred = model.predict(val_generator)
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        cm = confusion_matrix(Y_val, Y_pred_classes)
        plot_confusion_matrix(cm, target_names, os.path.join(fold_dir, 'confusion_matrix.png'))
        
        # Save Model
        model.save(os.path.join(args.output_dir, 'KFold_results', 'models', f'model_fold_{fold+1}.h5'))
        
        fold_results.append({
            'fold': fold+1,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'elapsed_time': elapsed_time
        })
        
        # Grad-CAM (on first 5 val images)
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer:
            for i in range(min(5, len(X_val))):
                img_path = os.path.join(fold_dir, f'gradcam_{i}.png')
                save_gradcam(model, np.expand_dims(X_val[i]/255.0, axis=0), X_val[i].astype(np.uint8), img_path, last_conv_layer)

    # Summary
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(args.output_dir, 'kfold_summary.csv'), index=False)
    print("\nK-Fold Results Summary:")
    print(df_results)
    print(f"Mean Accuracy: {df_results['val_acc'].mean():.4f} +/- {df_results['val_acc'].std():.4f}")

def run_single_training(args, X, Y, labels, target_names):
    """Executes a single training run (Train/Val/Test split)."""
    print("Starting Single Training Run...")
    
    # Split Data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=SEED, stratify=Y_train)
    
    # Preprocessing
    Y_train_cat = to_categorical(Y_train, num_classes=len(labels))
    Y_val_cat = to_categorical(Y_val, num_classes=len(labels))
    Y_test_cat = to_categorical(Y_test, num_classes=len(labels))
    
    datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=30, zoom_range=0.2,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow(X_train, Y_train_cat, batch_size=args.batch_size)
    val_generator = test_datagen.flow(X_val, Y_val_cat, batch_size=args.batch_size, shuffle=False)
    test_generator = test_datagen.flow(X_test, Y_test_cat, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = create_custom_model(args.img_width, args.img_height, len(labels))
    
    callbacks = [
        EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
        ReduceLROnPlateau(patience=5, monitor='val_loss', factor=0.2, verbose=1)
    ]
    
    history = model.fit(
        train_generator, 
        epochs=args.epochs, 
        validation_data=val_generator, 
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save Results
    save_training_plots(history, args.output_dir, prefix="single_run_")
    model.save(os.path.join(args.output_dir, 'MODELS', 'single_run_model.h5'))
    
    # Classification Report
    Y_pred = model.predict(test_generator)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    print(classification_report(Y_test, Y_pred_classes, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred_classes)
    plot_confusion_matrix(cm, target_names, os.path.join(args.output_dir, 'single_run_confusion_matrix.png'))

def run_transfer_learning(args, X, Y, labels, target_names):
    """Executes Transfer Learning experiments."""
    print("Starting Transfer Learning Experiments...")
    
    # Split Data (Same as single run)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=SEED, stratify=Y_train)
    
    Y_train_cat = to_categorical(Y_train, num_classes=len(labels))
    Y_val_cat = to_categorical(Y_val, num_classes=len(labels))
    Y_test_cat = to_categorical(Y_test, num_classes=len(labels))
    
    datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=30, zoom_range=0.2,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow(X_train, Y_train_cat, batch_size=args.batch_size)
    val_generator = test_datagen.flow(X_val, Y_val_cat, batch_size=args.batch_size, shuffle=False)
    test_generator = test_datagen.flow(X_test, Y_test_cat, batch_size=args.batch_size, shuffle=False)
    
    models_to_test = ["ResNet50", "VGG16", "MobileNetV2"] # Example subset
    results = []
    
    for base_model_name in models_to_test:
        print(f"\nTraining {base_model_name}...")
        model, base_model = create_transfer_model(base_model_name, args.img_width, args.img_height, len(labels))
        
        callbacks = [
            EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
            ReduceLROnPlateau(patience=5, monitor='val_loss', factor=0.2, verbose=1)
        ]
        
        start_time = time.time()
        history = model.fit(
            train_generator, 
            epochs=args.epochs, 
            validation_data=val_generator, 
            callbacks=callbacks,
            verbose=1
        )
        elapsed_time = time.time() - start_time
        
        test_loss, test_acc = model.evaluate(test_generator)
        results.append({
            'Model': base_model_name,
            'Test Accuracy': test_acc,
            'Test Loss': test_loss,
            'Time': elapsed_time
        })
        
        save_training_plots(history, args.output_dir, prefix=f"{base_model_name}_")
        model.save(os.path.join(args.output_dir, 'MODELS', f'{base_model_name}_finetuned.h5'))
        
        # Confusion Matrix
        Y_pred = model.predict(test_generator)
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        cm = confusion_matrix(Y_test, Y_pred_classes)
        plot_confusion_matrix(cm, target_names, os.path.join(args.output_dir, f'{base_model_name}_confusion_matrix.png'))
        
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'transfer_learning_results.csv'), index=False)
    
    # Generate comparison plot
    plot_model_comparison(results_df, os.path.join(args.output_dir, 'model_comparison.png'), 
                         title="Transfer Learning Models Comparison")
    
    # Generate text summary
    summary_path = os.path.join(args.output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSFER LEARNING TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Models Tested: {len(results)}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Image Size: {args.img_width}x{args.img_height}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RESULTS BY MODEL\n")
        f.write("-" * 80 + "\n\n")
        
        for idx, row in results_df.iterrows():
            f.write(f"Model: {row['Model']}\n")
            f.write(f"  Test Accuracy: {row['Test Accuracy']:.4f}\n")
            f.write(f"  Test Loss: {row['Test Loss']:.4f}\n")
            f.write(f"  Training Time: {row['Time']:.2f} seconds ({row['Time']/60:.2f} minutes)\n")
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Best Accuracy: {results_df['Test Accuracy'].max():.4f} ({results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']})\n")
        f.write(f"Lowest Loss: {results_df['Test Loss'].min():.4f} ({results_df.loc[results_df['Test Loss'].idxmin(), 'Model']})\n")
        f.write(f"Fastest Training: {results_df['Time'].min():.2f}s ({results_df.loc[results_df['Time'].idxmin(), 'Model']})\n")
        f.write(f"Mean Accuracy: {results_df['Test Accuracy'].mean():.4f} ± {results_df['Test Accuracy'].std():.4f}\n")
        f.write(f"Mean Loss: {results_df['Test Loss'].mean():.4f} ± {results_df['Test Loss'].std():.4f}\n")
        f.write(f"Total Training Time: {results_df['Time'].sum():.2f} seconds ({results_df['Time'].sum()/60:.2f} minutes)\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nTraining summary saved to {summary_path}")
    print("\nTransfer Learning Results:")
    print(results_df)

def main():
    args = parse_arguments()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
        
    setup_directories(args.output_dir)
    
    # Load Data
    labels_dict, subdirs, jpg_counts = get_data_info(args.data_dir)
    print(f"Found classes: {labels_dict}")
    print(f"Image counts: {jpg_counts}")
    
    target_names = [k for k, v in sorted(labels_dict.items(), key=lambda item: item[1])]
    
    X, Y = load_and_preprocess_images(args.data_dir, subdirs, args.img_width, args.img_height)
    print(f"Total images: {len(X)}")
    print(f"Data shape: {X.shape}")
    
    if args.mode == 'kfold':
        run_kfold_training(args, X, Y, labels_dict, target_names)
    elif args.mode == 'single':
        run_single_training(args, X, Y, labels_dict, target_names)
    elif args.mode == 'transfer':
        run_transfer_learning(args, X, Y, labels_dict, target_names)

if __name__ == "__main__":
    main()
