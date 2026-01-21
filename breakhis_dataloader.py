#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BreaKHis Dataset Loader and GAN-based Augmentation
===================================================
This module provides:
1. Efficient data loading from BreaKHis dataset
2. GAN-based synthetic image generation for dataset balancing
3. Advanced data augmentation strategies
4. Dataset statistics and analysis tools

Dataset Structure:
- Benign: adenosis, fibroadenoma, phyllodes_tumor, tubular_adenoma
- Malignant: ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma
- Magnifications: 40X, 100X, 200X, 400X
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json


class BreaKHisDataLoader:
    """
    Data loader for BreaKHis histopathological dataset
    """
    
    def __init__(self, base_path: str, image_size: Tuple[int, int] = (128, 128)):
        """
        Initialize data loader
        
        Args:
            base_path: Path to BreaKHis_v1/histology_slides/breast/
            image_size: Target image size (height, width)
        """
        self.base_path = Path(base_path)
        self.image_size = image_size
        
        # Class mappings
        self.benign_types = {
            'A': 'adenosis',
            'F': 'fibroadenoma',
            'PT': 'phyllodes_tumor',
            'TA': 'tubular_adenoma'
        }
        
        self.malignant_types = {
            'DC': 'ductal_carcinoma',
            'LC': 'lobular_carcinoma',
            'MC': 'mucinous_carcinoma',
            'PC': 'papillary_carcinoma'
        }
        
        self.magnifications = ['40X', '100X', '200X', '400X']
        
        # Binary classification
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        self.idx_to_class = {0: 'benign', 1: 'malignant'}
        
        # Multi-class classification (8 sub-types)
        self.subtype_to_idx = {}
        idx = 0
        for subtype in self.benign_types.values():
            self.subtype_to_idx[subtype] = idx
            idx += 1
        for subtype in self.malignant_types.values():
            self.subtype_to_idx[subtype] = idx
            idx += 1
        
        self.idx_to_subtype = {v: k for k, v in self.subtype_to_idx.items()}
        
        # Dataset statistics
        self.dataset_stats = None
    
    def parse_filename(self, filename: str) -> Dict:
        """
        Parse BreaKHis filename format
        
        Format: <BIOPSY>_<CLASS>_<TYPE>_<YEAR>-<SLIDE_ID>-<MAG>-<SEQ>.png
        Example: SOB_B_A-14-22549AB-100-001.png
        
        Args:
            filename: Image filename
            
        Returns:
            Dictionary with parsed information
        """
        parts = filename.replace('.png', '').split('-')
        
        prefix = parts[0].split('_')
        biopsy = prefix[0]  # SOB or CNB
        tumor_class = prefix[1]  # B (benign) or M (malignant)
        tumor_type = prefix[2]  # A, F, PT, TA, DC, LC, MC, PC
        
        year = parts[1]
        slide_id = parts[2]
        magnification = parts[3]
        sequence = parts[4]
        
        # Determine main class and subtype
        if tumor_class == 'B':
            main_class = 'benign'
            subtype = self.benign_types.get(tumor_type, 'unknown')
        else:
            main_class = 'malignant'
            subtype = self.malignant_types.get(tumor_type, 'unknown')
        
        return {
            'filename': filename,
            'biopsy': biopsy,
            'class': main_class,
            'class_idx': self.class_to_idx[main_class],
            'subtype': subtype,
            'subtype_idx': self.subtype_to_idx.get(subtype, -1),
            'year': year,
            'slide_id': slide_id,
            'magnification': magnification,
            'sequence': sequence
        }
    
    def load_dataset(self, magnification: Optional[str] = None,
                    binary: bool = True,
                    test_size: float = 0.3,
                    val_size: float = 0.15,
                    random_state: int = 42) -> Dict:
        """
        Load complete dataset
        
        Args:
            magnification: Specific magnification ('40X', '100X', '200X', '400X')
                          If None, loads all magnifications
            binary: If True, binary classification (benign vs malignant)
                   If False, 8-class classification (sub-types)
            test_size: Fraction for test set
            val_size: Fraction of training set for validation
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """
        print(f"Loading BreaKHis dataset from: {self.base_path}")
        print(f"Magnification: {magnification if magnification else 'All'}")
        print(f"Classification: {'Binary' if binary else 'Multi-class (8 subtypes)'}")
        
        # Collect all image paths and labels
        image_paths = []
        labels = []
        metadata = []
        
        # Load benign images
        benign_path = self.base_path / 'benign' / 'SOB'
        for subtype_name in self.benign_types.values():
            subtype_path = benign_path / subtype_name
            
            if not subtype_path.exists():
                print(f"Warning: {subtype_path} does not exist")
                continue
            
            # Iterate through all patient folders
            for patient_folder in subtype_path.iterdir():
                if not patient_folder.is_dir():
                    continue
                
                # Check for magnification folders
                mag_folders = [magnification] if magnification else self.magnifications
                
                for mag in mag_folders:
                    mag_folder = patient_folder / mag
                    
                    if not mag_folder.exists():
                        continue
                    
                    # Load all images in this magnification
                    for img_file in mag_folder.glob('*.png'):
                        image_paths.append(str(img_file))
                        
                        info = self.parse_filename(img_file.name)
                        
                        if binary:
                            labels.append(info['class_idx'])
                        else:
                            labels.append(info['subtype_idx'])
                        
                        metadata.append(info)
        
        # Load malignant images
        malignant_path = self.base_path / 'malignant' / 'SOB'
        for subtype_name in self.malignant_types.values():
            subtype_path = malignant_path / subtype_name
            
            if not subtype_path.exists():
                print(f"Warning: {subtype_path} does not exist")
                continue
            
            for patient_folder in subtype_path.iterdir():
                if not patient_folder.is_dir():
                    continue
                
                mag_folders = [magnification] if magnification else self.magnifications
                
                for mag in mag_folders:
                    mag_folder = patient_folder / mag
                    
                    if not mag_folder.exists():
                        continue
                    
                    for img_file in mag_folder.glob('*.png'):
                        image_paths.append(str(img_file))
                        
                        info = self.parse_filename(img_file.name)
                        
                        if binary:
                            labels.append(info['class_idx'])
                        else:
                            labels.append(info['subtype_idx'])
                        
                        metadata.append(info)
        
        print(f"\nTotal images loaded: {len(image_paths)}")
        
        # Convert to arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Print class distribution
        print("\nClass distribution:")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items()):
            if binary:
                class_name = self.idx_to_class[label]
            else:
                class_name = self.idx_to_subtype[label]
            print(f"  {class_name}: {count} images")
        
        # Split dataset
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test, meta_trainval, meta_test = \
            train_test_split(image_paths, labels, metadata, 
                           test_size=test_size,
                           stratify=labels,
                           random_state=random_state)
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, meta_train, meta_val = \
            train_test_split(X_trainval, y_trainval, meta_trainval,
                           test_size=val_size_adjusted,
                           stratify=y_trainval,
                           random_state=random_state)
        
        print(f"\nDataset split:")
        print(f"  Training: {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
        print(f"  Test: {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
        
        return {
            'train': {'paths': X_train, 'labels': y_train, 'metadata': meta_train},
            'val': {'paths': X_val, 'labels': y_val, 'metadata': meta_val},
            'test': {'paths': X_test, 'labels': y_test, 'metadata': meta_test},
            'num_classes': 2 if binary else 8,
            'class_names': list(self.class_to_idx.keys()) if binary else list(self.subtype_to_idx.keys())
        }
    
    def load_images_from_paths(self, image_paths: np.ndarray,
                              normalize: bool = True,
                              to_rgb: bool = True) -> np.ndarray:
        """
        Load images from file paths
        
        Args:
            image_paths: Array of image file paths
            normalize: Whether to normalize to [0, 1]
            to_rgb: Whether to convert to RGB
            
        Returns:
            NumPy array of images
        """
        images = []
        
        for path in image_paths:
            img = cv2.imread(path)
            
            if to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.image_size)
            
            if normalize:
                img = img.astype('float32') / 255.0
            
            images.append(img)
        
        return np.array(images)
    
    def create_tf_dataset(self, image_paths: np.ndarray, labels: np.ndarray,
                         batch_size: int = 32, shuffle: bool = True,
                         augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow Dataset
        
        Args:
            image_paths: Image file paths
            labels: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle
            augment: Whether to apply augmentation
            
        Returns:
            tf.data.Dataset
        """
        def load_and_preprocess(path, label):
            # Load image
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, self.image_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            if augment:
                # Random augmentation
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.random_brightness(img, max_delta=0.1)
                img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
                img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
            
            return img, label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        dataset = dataset.map(load_and_preprocess, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def analyze_dataset(self, dataset_dict: Dict, save_dir: str = 'dataset_analysis'):
        """
        Analyze and visualize dataset statistics
        
        Args:
            dataset_dict: Dataset dictionary from load_dataset()
            save_dir: Directory to save analysis plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("Dataset Analysis")
        print("="*70)
        
        # 1. Class distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, split in enumerate(['train', 'val', 'test']):
            labels = dataset_dict[split]['labels']
            
            if dataset_dict['num_classes'] == 2:
                class_names = ['Benign', 'Malignant']
            else:
                class_names = dataset_dict['class_names']
            
            unique, counts = np.unique(labels, return_counts=True)
            
            axes[idx].bar(range(len(unique)), counts, color=['green' if dataset_dict['num_classes']==2 and i==0 else 'red' for i in unique])
            axes[idx].set_xlabel('Class', fontsize=12)
            axes[idx].set_ylabel('Number of Images', fontsize=12)
            axes[idx].set_title(f'{split.capitalize()} Set Distribution', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(range(len(unique)))
            axes[idx].set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
            
            # Add value labels on bars
            for i, count in enumerate(counts):
                axes[idx].text(i, count, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Magnification distribution (if metadata available)
        if dataset_dict['train']['metadata']:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, split in enumerate(['train', 'val', 'test']):
                metadata = dataset_dict[split]['metadata']
                magnifications = [m['magnification'] for m in metadata]
                
                mag_counts = Counter(magnifications)
                
                axes[idx].bar(mag_counts.keys(), mag_counts.values(), color='steelblue')
                axes[idx].set_xlabel('Magnification', fontsize=12)
                axes[idx].set_ylabel('Number of Images', fontsize=12)
                axes[idx].set_title(f'{split.capitalize()} Set - Magnification', fontsize=14, fontweight='bold')
                
                for mag, count in mag_counts.items():
                    axes[idx].text(mag, count, str(count), ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'magnification_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Sample images
        train_paths = dataset_dict['train']['paths']
        train_labels = dataset_dict['train']['labels']
        
        # Show samples from each class
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        for class_idx in range(min(dataset_dict['num_classes'], 2)):  # Show up to 2 classes
            # Get indices for this class
            class_indices = np.where(train_labels == class_idx)[0]
            
            # Sample 5 images
            sample_indices = np.random.choice(class_indices, size=min(5, len(class_indices)), replace=False)
            
            for i, idx in enumerate(sample_indices):
                img_path = train_paths[idx]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                ax_idx = class_idx * 5 + i
                axes[ax_idx].imshow(img)
                
                if dataset_dict['num_classes'] == 2:
                    class_name = self.idx_to_class[class_idx]
                else:
                    class_name = self.idx_to_subtype[class_idx]
                
                axes[ax_idx].set_title(f'{class_name}', fontsize=10, fontweight='bold')
                axes[ax_idx].axis('off')
        
        # Hide unused subplots
        for i in range(dataset_dict['num_classes'] * 5, 10):
            axes[i].axis('off')
        
        plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAnalysis plots saved to: {save_dir}/")
        
        # Save statistics to JSON
        stats = {
            'total_images': sum([len(dataset_dict[split]['paths']) for split in ['train', 'val', 'test']]),
            'train_size': len(dataset_dict['train']['paths']),
            'val_size': len(dataset_dict['val']['paths']),
            'test_size': len(dataset_dict['test']['paths']),
            'num_classes': dataset_dict['num_classes'],
            'class_names': dataset_dict['class_names']
        }
        
        for split in ['train', 'val', 'test']:
            labels = dataset_dict[split]['labels']
            unique, counts = np.unique(labels, return_counts=True)
            stats[f'{split}_distribution'] = {int(u): int(c) for u, c in zip(unique, counts)}
        
        with open(os.path.join(save_dir, 'dataset_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to: {save_dir}/dataset_statistics.json")


class ConditionalGAN:
    """
    Conditional GAN for generating synthetic histopathological images
    
    This helps balance the dataset and provides additional training samples
    for minority classes.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (128, 128),
                 num_classes: int = 2,
                 latent_dim: int = 100):
        """
        Initialize Conditional GAN
        
        Args:
            image_size: Size of generated images
            num_classes: Number of classes
            latent_dim: Dimension of latent space
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Build generator and discriminator
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Optimizers
        self.generator_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
        
    def _build_generator(self) -> keras.Model:
        """Build generator network"""
        # Noise input
        noise = layers.Input(shape=(self.latent_dim,))
        
        # Class label input
        label = layers.Input(shape=(1,), dtype='int32')
        label_embedding = layers.Embedding(self.num_classes, 50)(label)
        label_embedding = layers.Flatten()(label_embedding)
        
        # Concatenate noise and label
        gen_input = layers.Concatenate()([noise, label_embedding])
        
        # Dense layers
        x = layers.Dense(8 * 8 * 256, use_bias=False)(gen_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((8, 8, 256))(x)
        
        # Upsampling blocks
        x = layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(32, 5, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Conv2DTranspose(3, 5, strides=2, padding='same', use_bias=False, activation='tanh')(x)
        
        model = keras.Model([noise, label], x, name='generator')
        return model
    
    def _build_discriminator(self) -> keras.Model:
        """Build discriminator network"""
        # Image input
        img = layers.Input(shape=(*self.image_size, 3))
        
        # Class label input
        label = layers.Input(shape=(1,), dtype='int32')
        label_embedding = layers.Embedding(self.num_classes, 50)(label)
        label_embedding = layers.Flatten()(label_embedding)
        label_embedding = layers.Dense(self.image_size[0] * self.image_size[1])(label_embedding)
        label_embedding = layers.Reshape((*self.image_size, 1))(label_embedding)
        
        # Concatenate image and label
        disc_input = layers.Concatenate()([img, label_embedding])
        
        # Convolutional blocks
        x = layers.Conv2D(64, 5, strides=2, padding='same')(disc_input)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)
        
        model = keras.Model([img, label], x, name='discriminator')
        return model
    
    def generate_synthetic_images(self, num_images: int, class_label: int) -> np.ndarray:
        """
        Generate synthetic images for a specific class
        
        Args:
            num_images: Number of images to generate
            class_label: Class label for conditional generation
            
        Returns:
            Generated images
        """
        noise = tf.random.normal([num_images, self.latent_dim])
        labels = tf.ones((num_images, 1), dtype=tf.int32) * class_label
        
        generated_images = self.generator([noise, labels], training=False)
        
        # Convert from [-1, 1] to [0, 1]
        generated_images = (generated_images + 1) / 2.0
        
        return generated_images.numpy()


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("BreaKHis Dataset Loader and GAN Augmentation")
    print("="*70)
    
    # Example: Load dataset
    base_path = "BreaKHis_v1/histology_slides/breast"
    
    loader = BreaKHisDataLoader(base_path, image_size=(128, 128))
    
    print("\nExample usage:")
    print("""
    # Load dataset for 200X magnification
    dataset = loader.load_dataset(
        magnification='200X',
        binary=True,
        test_size=0.3,
        val_size=0.15
    )
    
    # Analyze dataset
    loader.analyze_dataset(dataset, save_dir='dataset_analysis')
    
    # Load actual images
    train_images = loader.load_images_from_paths(
        dataset['train']['paths']
    )
    
    # Create TensorFlow dataset
    train_dataset = loader.create_tf_dataset(
        dataset['train']['paths'],
        dataset['train']['labels'],
        batch_size=32,
        augment=True
    )
    """)
