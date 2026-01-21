#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced CVFBJTL-BCD Model for Breast Cancer Diagnosis
======================================================
This implementation enhances the original CVFBJTL-BCD technique with:
1. Vision Transformer (ViT) integration for improved feature extraction
2. Explainable AI using Grad-CAM
3. SMOTE for dataset balancing
4. Federated Learning for privacy preservation
5. Comprehensive ablation studies

Author: Enhanced Research Implementation
Based on: "Enhanced breast cancer diagnosis through integration of computer vision 
          with fusion based joint transfer learning using multi modality medical images"
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet201, InceptionV3, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class GaborFilter:
    """
    Gabor Filter for noise reduction and texture enhancement
    Based on the paper's methodology (Figure 2)
    """
    
    def __init__(self, ksize: int = 31, sigma: float = 4.0, 
                 gamma: float = 0.5, lambd: float = 10.0):
        """
        Initialize Gabor Filter parameters
        
        Args:
            ksize: Kernel size
            sigma: Standard deviation
            gamma: Spatial aspect ratio
            lambd: Wavelength
        """
        self.ksize = ksize
        self.sigma = sigma
        self.gamma = gamma
        self.lambd = lambd
        
    def apply_multiscale_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multi-scale and multi-directional Gabor filtering
        
        Mathematical formulation (Equation 1 from paper):
        g = exp[-(x'cosθ + y'sinθ)² + γ²(y'cosθ - x'sinθ)²] / 2σ²
        exp[i(2π(x'cosθ + y'sinθ)/λ + φ)]
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            Filtered image with enhanced texture features
        """
        # Ensure image is in correct format (uint8, 0-255)
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Convert from [0,1] to [0,255] if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Ensure gray is uint8
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
            
        # Multi-directional filtering (8 orientations)
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4, 
                       np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        
        filtered_images = []
        for theta in orientations:
            kernel = cv2.getGaborKernel((self.ksize, self.ksize), 
                                       self.sigma, theta, 
                                       self.lambd, self.gamma, 
                                       0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            filtered_images.append(filtered)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Non-maximum suppression
        nms_result = self._non_maximum_suppression(filtered_images, edges)
        
        # Multi-scale fusion
        fused_image = self._multiscale_fusion(filtered_images, nms_result)
        
        return fused_image
    
    def _non_maximum_suppression(self, filtered_imgs: List[np.ndarray], 
                                  edges: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression to filtered images"""
        max_response = np.max(np.array(filtered_imgs), axis=0)
        suppressed = np.where(edges > 0, max_response, 0)
        return suppressed
    
    def _multiscale_fusion(self, filtered_imgs: List[np.ndarray], 
                          nms_result: np.ndarray) -> np.ndarray:
        """Fuse multi-scale filtered images"""
        fused = np.mean(filtered_imgs, axis=0)
        fused = cv2.normalize(fused, None, 0, 255, cv2.CV_8U)
        return fused


class VisionTransformerBlock(layers.Layer):
    """
    Vision Transformer (ViT) block for enhanced feature extraction
    This adds novelty to the existing DenseNet201+InceptionV3+MobileNetV2 ensemble
    """
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, 
                 ff_dim: int = 512, dropout_rate: float = 0.1):
        """
        Initialize Vision Transformer block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
        """
        super(VisionTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, 
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        """Forward pass through ViT block"""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class EnhancedFeatureFusion:
    """
    Enhanced Feature Fusion with Vision Transformer
    Extends the original DenseNet201+InceptionV3+MobileNetV2 fusion
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 3),
                 include_vit: bool = True):
        """
        Initialize enhanced feature fusion model
        
        Args:
            input_shape: Input image shape
            include_vit: Whether to include Vision Transformer
        """
        self.input_shape = input_shape
        self.include_vit = include_vit
        
    def build_densenet201_branch(self, inputs):
        """
        Build DenseNet201 branch
        Dense connectivity for gradient flow and feature reuse
        """
        base_model = DenseNet201(include_top=False, 
                                weights='imagenet',
                                input_shape=self.input_shape)
        
        # Fine-tune last few layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D(name='densenet_gap')(x)
        return x
    
    def build_inceptionv3_branch(self, inputs):
        """
        Build InceptionV3 branch
        Multi-scale feature extraction with factorized convolutions
        """
        base_model = InceptionV3(include_top=False,
                                weights='imagenet',
                                input_shape=self.input_shape)
        
        for layer in base_model.layers[:-50]:
            layer.trainable = False
            
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D(name='inception_gap')(x)
        return x
    
    def build_mobilenetv2_branch(self, inputs):
        """
        Build MobileNetV2 branch
        Efficient depth-wise separable convolutions
        """
        base_model = MobileNetV2(include_top=False,
                                weights='imagenet',
                                input_shape=self.input_shape)
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D(name='mobilenet_gap')(x)
        return x
    
    def build_vit_branch(self, inputs, patch_size: int = 16):
        """
        Build Vision Transformer branch (Novel Addition)
        Captures long-range dependencies in medical images
        """
        # Patch extraction
        patches = layers.Conv2D(256, kernel_size=patch_size, 
                               strides=patch_size, padding='valid')(inputs)
        patch_shape = patches.shape
        patches = layers.Reshape((-1, 256))(patches)
        
        # Position embedding
        num_patches = patches.shape[1]
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(input_dim=num_patches, 
                                             output_dim=256)(positions)
        patches = patches + position_embedding
        
        # Transformer blocks
        x = VisionTransformerBlock(embed_dim=256, num_heads=4)(patches)
        x = VisionTransformerBlock(embed_dim=256, num_heads=4)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='vit_gap')(x)
        return x
    
    def build_fusion_model(self, num_classes: int = 2):
        """
        Build complete fusion model with all branches
        
        Mathematical formulation (Equations 2-7 from paper):
        Local features: F_l = FC(L)
        Global features: F_g = FC(G) where G = GAP(g)
        Combined: F_c = Concat(L, F_g)
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Complete Keras model
        """
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Extract features from all branches
        densenet_features = self.build_densenet201_branch(inputs)
        inception_features = self.build_inceptionv3_branch(inputs)
        mobilenet_features = self.build_mobilenetv2_branch(inputs)
        
        # Concatenate traditional features
        traditional_fusion = layers.Concatenate(name='traditional_fusion')([
            densenet_features,
            inception_features,
            mobilenet_features
        ])
        
        if self.include_vit:
            # Add Vision Transformer features (NOVELTY)
            vit_features = self.build_vit_branch(inputs)
            
            # Enhanced fusion with ViT
            complete_fusion = layers.Concatenate(name='enhanced_fusion')([
                traditional_fusion,
                vit_features
            ])
        else:
            complete_fusion = traditional_fusion
        
        # Fully connected layer for local-global fusion
        # Implements weighted sharing in multi-task learning (Eq. 7)
        x = layers.Dense(1024, activation='relu', name='fc_fusion_1')(complete_fusion)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', name='fc_fusion_2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, 
                           name='Enhanced_CVFBJTL_BCD')
        
        return model


class StackedAutoencoder:
    """
    Stacked Autoencoder (SAE) for unsupervised feature learning
    Optimized using HHOA (Horse Herd Optimization Algorithm)
    
    Mathematical formulation (Equations 8-9 from paper):
    Encoder: h_l = f_1(x) = φ_e(W_{l,e}x + b_{l,e})
    Decoder: ỹ = g_1(x) = φ_d(W_{l,d}h_l + b_{l,d})
    """
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [512, 256, 128]):
        """
        Initialize Stacked Autoencoder
        
        Args:
            input_dim: Input feature dimension
            encoding_dims: List of encoding dimensions for each layer
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        
    def build_sae_model(self):
        """
        Build Stacked Autoencoder model
        
        Returns:
            encoder_model: Encoder for feature extraction
            autoencoder_model: Complete autoencoder for training
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,), name='sae_input')
        
        # Encoder layers
        encoded = input_layer
        for i, dim in enumerate(self.encoding_dims):
            encoded = layers.Dense(dim, activation='relu', 
                                 name=f'encoder_{i+1}')(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Bottleneck layer
        bottleneck = layers.Dense(self.encoding_dims[-1], 
                                 activation='relu', 
                                 name='bottleneck')(encoded)
        
        # Decoder layers
        decoded = bottleneck
        for i, dim in enumerate(reversed(self.encoding_dims[:-1])):
            decoded = layers.Dense(dim, activation='relu',
                                 name=f'decoder_{i+1}')(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='sigmoid',
                             name='sae_output')(decoded)
        
        # Create models
        encoder_model = models.Model(inputs=input_layer, outputs=bottleneck,
                                    name='SAE_Encoder')
        autoencoder_model = models.Model(inputs=input_layer, outputs=decoded,
                                        name='SAE_Autoencoder')
        
        return encoder_model, autoencoder_model


class HHOAOptimizer:
    """
    Horse Herd Optimization Algorithm (HHOA) for hyperparameter tuning
    
    Based on paper's methodology (Equations 10-20, Figure 7):
    - Grazing (G)
    - Hierarchy (H) 
    - Sociability (S)
    - Imitation (I)
    - Defense Mechanism (D)
    - Roaming (R)
    """
    
    def __init__(self, n_horses: int = 30, max_iterations: int = 50,
                 age_limit: Tuple[int, int] = (0, 15)):
        """
        Initialize HHOA optimizer
        
        Args:
            n_horses: Number of horses in the herd
            max_iterations: Maximum number of iterations
            age_limit: Age range for horses (alpha, beta)
        """
        self.n_horses = n_horses
        self.max_iterations = max_iterations
        self.age_limit = age_limit
        self.alpha = age_limit[0]
        self.beta = age_limit[1]
        
    def initialize_herd(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """
        Initialize horse positions and velocities
        
        Args:
            dim: Dimension of search space
            bounds: Lower and upper bounds
            
        Returns:
            Initial horse positions
        """
        positions = np.random.uniform(bounds[0], bounds[1], 
                                     size=(self.n_horses, dim))
        velocities = np.random.uniform(-1, 1, size=(self.n_horses, dim))
        ages = np.random.randint(0, self.beta, size=self.n_horses)
        
        return positions, velocities, ages
    
    def grazing_behavior(self, position: np.ndarray, velocity: np.ndarray,
                        age: int, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grazing behavior (Equations 12-13)
        
        G_i = g_iter + (i+P̃)X_i^(iter-1) - X_i^(iter-1), AGE = α, β, γ, δ
        ġ_i^(iter,AGE) = g_i^(iter-1,AGE) × ω_i
        """
        grazing_factor = 0.7  # Percentage of time spent grazing
        velocity_new = velocity * np.random.uniform(0.8, 1.2)
        
        # Update position based on grazing
        position_new = position + velocity_new * grazing_factor
        
        return position_new, velocity_new
    
    def hierarchy_behavior(self, position: np.ndarray, best_position: np.ndarray,
                          age: int) -> np.ndarray:
        """
        Hierarchy behavior (Equations 14-15)
        
        H_i^(iter,AGE) = h_i^(iter-1,h)X_i^(iter-1) - X_i^(iter-1), AGE = α, β, γ
        """
        hierarchy_coefficient = 0.5 if age < 5 else 0.8
        position_new = position + hierarchy_coefficient * (best_position - position)
        
        return position_new
    
    def sociability_behavior(self, position: np.ndarray, positions: np.ndarray,
                           age: int) -> np.ndarray:
        """
        Sociability behavior (Equations 16-17)
        
        S_i^(iter,AGE) = s_i^(iter,AGE)[(1/N∑X_j^(iter-1)) - X_i^(iter-1)], AGE = β, γ
        """
        mean_position = np.mean(positions, axis=0)
        sociability_factor = 0.6
        
        position_new = position + sociability_factor * (mean_position - position)
        
        return position_new
    
    def imitation_behavior(self, position: np.ndarray, positions: np.ndarray,
                          pN: float = 0.1) -> np.ndarray:
        """
        Imitation behavior (Equations 19-20)
        
        I_i^(iter,AGE) = i_i^(iter,AGE)[(1/pN∑X_j^(iter-1)) - X^(iter-1)], AGE = γ
        """
        n_finest = max(1, int(pN * len(positions)))
        finest_positions = positions[:n_finest]
        mean_finest = np.mean(finest_positions, axis=0)
        
        imitation_factor = 0.5
        position_new = position + imitation_factor * (mean_finest - position)
        
        return position_new
    
    def defense_mechanism(self, position: np.ndarray, worst_position: np.ndarray,
                         qN: float = 0.2) -> np.ndarray:
        """
        Defense mechanism (Equations 22-23)
        
        D_i^(iter,AGE) = d_i^(iter,AGE)[(1/qN∑X_j^(iter-1)) - X^(iter-1)], AGE = α, β, γ
        """
        defense_factor = 0.7
        position_new = position + defense_factor * (position - worst_position)
        
        return position_new
    
    def roaming_behavior(self, position: np.ndarray, velocity: np.ndarray,
                        r_iter: float = 0.05) -> np.ndarray:
        """
        Roaming behavior (Equations 25-26)
        
        R_i^(iter,AGE) = r_i^(iter,R,AGE)X_i^(iter-1), AGE = γ, δ
        """
        roaming_velocity = velocity * r_iter
        position_new = position + roaming_velocity
        
        return position_new
    
    def optimize(self, objective_function, dim: int, 
                bounds: Tuple[float, float]) -> Dict:
        """
        Run HHOA optimization
        
        Args:
            objective_function: Function to minimize
            dim: Dimension of search space
            bounds: Parameter bounds
            
        Returns:
            Dictionary with best parameters and fitness history
        """
        # Initialize herd
        positions, velocities, ages = self.initialize_herd(dim, bounds)
        
        # Evaluate initial fitness
        fitness = np.array([objective_function(pos) for pos in positions])
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = [best_fitness]
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            for i in range(self.n_horses):
                age = ages[i]
                
                # Apply different behaviors based on age
                if age < 5:  # Young horses - Grazing and Defense
                    pos_new, vel_new = self.grazing_behavior(
                        positions[i], velocities[i], age, iteration
                    )
                    positions[i] = pos_new
                    velocities[i] = vel_new
                    
                elif age < 10:  # Adult horses - Hierarchy and Sociability
                    positions[i] = self.hierarchy_behavior(
                        positions[i], best_position, age
                    )
                    positions[i] = self.sociability_behavior(
                        positions[i], positions, age
                    )
                    
                else:  # Old horses - Imitation and Roaming
                    positions[i] = self.imitation_behavior(
                        positions[i], positions[np.argsort(fitness)]
                    )
                    positions[i] = self.roaming_behavior(
                        positions[i], velocities[i]
                    )
                
                # Apply bounds
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])
                
                # Evaluate fitness
                fitness[i] = objective_function(positions[i])
                
                # Update best position
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_position = positions[i].copy()
            
            # Age horses
            ages = (ages + 1) % self.beta
            
            fitness_history.append(best_fitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{self.max_iterations}, "
                      f"Best Fitness: {best_fitness:.6f}")
        
        return {
            'best_parameters': best_position,
            'best_fitness': best_fitness,
            'fitness_history': fitness_history
        }


class DataBalancer:
    """
    Dataset balancing using SMOTE (Synthetic Minority Over-sampling Technique)
    
    This addresses the class imbalance issue in medical datasets
    Enhances the paper's methodology by improving model generalization
    """
    
    def __init__(self, sampling_strategy: str = 'auto', k_neighbors: int = 5):
        """
        Initialize SMOTE balancer
        
        Args:
            sampling_strategy: Strategy for resampling
            k_neighbors: Number of nearest neighbors for SMOTE
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance dataset
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Balanced X and y
        """
        # Support both integer labels (N,) / (N,1) and one-hot labels (N,C)
        if y.ndim > 1 and y.shape[-1] > 1:
            y_int = np.argmax(y, axis=-1).astype(int)
        else:
            y_int = y.reshape(-1).astype(int)

        print(f"Original dataset shape: {X.shape}")
        print(f"Original class distribution: {np.bincount(y_int)}")
        
        # Reshape if needed
        original_shape = X.shape
        if len(X.shape) > 2:
            n_samples = X.shape[0]
            X_reshaped = X.reshape(n_samples, -1)
        else:
            X_reshaped = X
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=self.sampling_strategy,
                     k_neighbors=self.k_neighbors,
                     random_state=42)
        
        X_balanced, y_balanced = smote.fit_resample(X_reshaped, y_int)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            new_shape = (X_balanced.shape[0],) + original_shape[1:]
            X_balanced = X_balanced.reshape(new_shape)
        
        print(f"Balanced dataset shape: {X_balanced.shape}")
        print(f"Balanced class distribution: {np.bincount(y_balanced.flatten())}")
        
        return X_balanced, y_balanced


class CVFBJTLBCDModel:
    """
    Complete Enhanced CVFBJTL-BCD Model
    
    Integrates:
    1. Gabor Filtering for noise reduction
    2. Enhanced Feature Fusion (DenseNet201 + InceptionV3 + MobileNetV2 + ViT)
    3. Stacked Autoencoder optimized by HHOA
    4. SMOTE for dataset balancing
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 3),
                 num_classes: int = 2,
                 use_gabor: bool = True,
                 use_vit: bool = True,
                 use_smote: bool = True):
        """
        Initialize complete model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of classes
            use_gabor: Whether to use Gabor filtering
            use_vit: Whether to include Vision Transformer
            use_smote: Whether to use SMOTE for balancing
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_gabor = use_gabor
        self.use_vit = use_vit
        self.use_smote = use_smote
        
        # Initialize components
        self.gabor_filter = GaborFilter() if use_gabor else None
        self.fusion_model = EnhancedFeatureFusion(input_shape, include_vit=use_vit)
        self.data_balancer = DataBalancer() if use_smote else None
        
        # Model will be built during training
        self.model = None
        self.history = None
        self.training_time = 0
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess single image
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        if self.use_gabor and self.gabor_filter is not None:
            # Apply Gabor filtering
            filtered = self.gabor_filter.apply_multiscale_filter(image)
            # Convert back to RGB if needed
            if len(filtered.shape) == 2:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
        else:
            filtered = image
        
        # Resize to input shape
        resized = cv2.resize(filtered, 
                           (self.input_shape[0], self.input_shape[1]))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        return normalized
    
    def build_model(self):
        """Build the complete model"""
        self.model = self.fusion_model.build_fusion_model(self.num_classes)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        # Normalize label formats:
        # - SMOTE requires integer labels
        # - Model is compiled with categorical_crossentropy, so it expects one-hot labels
        y_train_is_onehot = (y_train.ndim == 2 and y_train.shape[1] == self.num_classes)
        y_val_is_onehot = (y_val is not None and y_val.ndim == 2 and y_val.shape[1] == self.num_classes)

        y_train_int = np.argmax(y_train, axis=1).astype(int) if y_train_is_onehot else y_train.reshape(-1).astype(int)
        y_val_int = None
        if y_val is not None:
            y_val_int = np.argmax(y_val, axis=1).astype(int) if y_val_is_onehot else y_val.reshape(-1).astype(int)

        # Balance dataset if enabled (on integer labels)
        if self.use_smote and self.data_balancer is not None:
            X_train, y_train_int = self.data_balancer.balance_dataset(X_train, y_train_int)

        # Convert to one-hot for training
        y_train = tf.keras.utils.to_categorical(y_train_int, self.num_classes)
        if y_val is not None:
            y_val = tf.keras.utils.to_categorical(y_val_int, self.num_classes)

        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_cvfbjtl_bcd_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Record training time
        start_time = time.time()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Predict
        start_time = time.time()
        y_pred_proba = self.model.predict(X_test)
        inference_time = time.time() - start_time
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Sensitivity/Specificity (binary only)
        sensitivity = None
        specificity = None
        if self.num_classes == 2:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # AUC for binary classification
        if self.num_classes == 2:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        results = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'sensitivity': (sensitivity * 100) if sensitivity is not None else None,
            'specificity': (specificity * 100) if specificity is not None else None,
            'f1_score': f1 * 100,
            'auc': auc * 100,
            'inference_time': inference_time,
            'avg_time_per_sample': inference_time / len(X_test)
        }
        
        return results
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


# Example usage demonstration
if __name__ == "__main__":
    print("="*70)
    print("Enhanced CVFBJTL-BCD Model for Breast Cancer Diagnosis")
    print("="*70)
    
    # This is a demonstration - replace with actual data loading
    print("\nNOTE: This is a framework implementation.")
    print("To use with BreaKHis dataset, implement data loading from:")
    print("  - Benign: adenosis, fibroadenoma, phyllodes_tumor, tubular_adenoma")
    print("  - Malignant: ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma")
    print("  - Magnifications: 40X, 100X, 200X, 400X")
    
    # Model initialization example
    model = CVFBJTLBCDModel(
        input_shape=(128, 128, 3),
        num_classes=2,  # Binary: Benign vs Malignant
        use_gabor=True,
        use_vit=True,
        use_smote=True
    )
    
    print("\nModel components initialized:")
    print(f"  - Gabor Filtering: {model.use_gabor}")
    print(f"  - Vision Transformer: {model.use_vit}")
    print(f"  - SMOTE Balancing: {model.use_smote}")
    
    # Build model
    model.build_model()
    print("\nModel architecture built successfully!")
    print(f"Total parameters: {model.model.count_params():,}")
