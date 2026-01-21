import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Detect Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle/input')
KAGGLE_WORKING = '/kaggle/working' if IS_KAGGLE else '.'
KAGGLE_INPUT = '/kaggle/input' if IS_KAGGLE else 'BreaKHis_v1/histology_slides/breast'

print(f"🔍 Environment Detection:")
print(f"   Running on Kaggle: {IS_KAGGLE}")
print(f"   Working Directory: {KAGGLE_WORKING}")
print(f"   Input Directory: {KAGGLE_INPUT}")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import gc

# Import custom modules
import importlib
import enhanced_cvfbjtl_bcd_model
importlib.reload(enhanced_cvfbjtl_bcd_model)
from enhanced_cvfbjtl_bcd_model import (
    CVFBJTLBCDModel, GaborFilter, EnhancedFeatureFusion, 
    StackedAutoencoder, HHOAOptimizer, DataBalancer
)
from breakhis_dataloader import BreaKHisDataLoader
from advanced_explainability import GradCAMPlusPlus as GradCAM

# ============================================================================
# GPU Configuration for Kaggle
# ============================================================================

def setup_kaggle_gpu():
    """
    Configure GPU settings optimized for Kaggle environment
    Kaggle provides: Tesla P100 (16GB) or T4 (16GB)
    """
    print("\n" + "="*70)
    print("🚀 GPU CONFIGURATION FOR KAGGLE")
    print("="*70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth (prevents OOM errors)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision training (faster on Kaggle GPUs)
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            
            print(f"✅ Found {len(gpus)} GPU(s)")
            print(f"   GPU Name: {gpus[0].name}")
            print(f"   Mixed Precision: ENABLED (float16)")
            print(f"   Memory Growth: ENABLED")
            
            # XLA optimization (JIT compilation)
            tf.config.optimizer.set_jit(True)
            print(f"   XLA JIT Compilation: ENABLED")
            
            return True
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
            return False
    else:
        print("⚠️  No GPU found. Training will be VERY SLOW on CPU.")
        print("   Kaggle notebooks should have GPU enabled by default.")
        print("   Go to: Settings → Accelerator → GPU (T4 x2)")
        return False


def optimize_tensorflow_for_kaggle():
    """
    Apply TensorFlow optimizations for Kaggle environment
    """
    # Thread optimization
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    # Auto-tuning for data pipeline
    tf.data.experimental.enable_debug_mode()
    
    print("✅ TensorFlow optimizations applied")


# ============================================================================
# Enhanced Configuration for Better Performance
# ============================================================================

class KaggleTrainingConfig:
    """
    Enhanced configuration to exceed paper's performance
    
    Paper Results:
    - Histopathological: 98.18% accuracy
    - Ultrasound: 99.15% accuracy
    
    Target: >98.5% on Histopathological (BreaKHis)
    """
    
    def __init__(self):
        # Data paths (auto-detect Kaggle or local)
        if IS_KAGGLE:
            # You need to upload BreaKHis dataset to Kaggle
            # Expected path: /kaggle/input/breakhis-dataset/...
            self.data_path = self._find_kaggle_dataset()
        else:
            self.data_path = 'BreaKHis_v1/histology_slides/breast'
        
        # Output directory
        self.output_dir = os.path.join(KAGGLE_WORKING, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Image settings
        self.image_size = 128  # Paper uses 128x128
        self.magnification = '200X'  # Best performance in paper
        self.binary = True  # Benign vs Malignant
        
        # Model architecture
        self.use_gabor = True  # Gabor filtering (paper's method)
        self.use_vit = True  # Vision Transformer (NOVELTY)
        self.use_smote = True  # SMOTE balancing (ENHANCEMENT)
        self.use_sae = True  # Stacked Autoencoder (paper's method)
        self.use_hhoa = False  # HHOA takes too long on Kaggle (50 iters)
        
        # Training hyperparameters (optimized for Kaggle)
        self.epochs = 50  # Paper uses 50 epochs
        self.batch_size = 32  # Balanced for 16GB GPU
        self.learning_rate = 0.0001  # Paper's learning rate
        
        # Advanced settings
        self.use_augmentation = True
        self.use_early_stopping = True
        self.patience = 10
        
        # Callbacks
        self.use_reduce_lr = True
        self.reduce_lr_patience = 5
        self.reduce_lr_factor = 0.5
        
        # Reproducibility
        self.seed = 42
        
    def _find_kaggle_dataset(self):
        """
        Auto-detect BreaKHis dataset path in Kaggle input
        """
        if not IS_KAGGLE:
            return None
        
        # Dataset from: https://www.kaggle.com/datasets/ambarish/breakhis
        # In Kaggle notebook, add this dataset to input (+ Add Data button)
        possible_names = [
            'breakhis',  # Primary: ambarish/breakhis
            'breakhis-dataset',
            'breast-cancer-histopathological',
            'breakhis-v1'
        ]
        
        for name in possible_names:
            path = f'/kaggle/input/{name}'
            if os.path.exists(path):
                print(f"✅ Found dataset: {path}")
                # Find the actual data directory
                for root, dirs, files in os.walk(path):
                    if 'benign' in dirs and 'malignant' in dirs:
                        print(f"   Data directory: {root}")
                        return root
        
        print("⚠️  BreaKHis dataset not found in Kaggle input!")
        print("   Add dataset: https://www.kaggle.com/datasets/ambarish/breakhis")
        print("   Click '+ Add Data' in Kaggle notebook and search 'ambarish breakhis'")
        print("   1. Go to: https://www.kaggle.com/datasets")
        print("   2. Create New Dataset")
        print("   3. Upload BreaKHis_v1 folder")
        print("   4. Add dataset to this notebook")
        
        return '/kaggle/input/breakhis-dataset'
    
    def set_seeds(self):
        """Set all random seeds for reproducibility"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("⚙️  TRAINING CONFIGURATION")
        print("="*70)
        print(f"📁 Data Path: {self.data_path}")
        print(f"📊 Magnification: {self.magnification}")
        print(f"🖼️  Image Size: {self.image_size}x{self.image_size}")
        print(f"🎯 Task: {'Binary' if self.binary else 'Multi-class'} Classification")
        print(f"\n🏗️  Architecture:")
        print(f"   Gabor Filtering: {'✓' if self.use_gabor else '✗'}")
        print(f"   Vision Transformer: {'✓' if self.use_vit else '✗'}")
        print(f"   SMOTE Balancing: {'✓' if self.use_smote else '✗'}")
        print(f"   Stacked Autoencoder: {'✓' if self.use_sae else '✗'}")
        print(f"   HHOA Optimization: {'✓' if self.use_hhoa else '✗'}")
        print(f"\n🎓 Training:")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Early Stopping: {'✓' if self.use_early_stopping else '✗'}")
        print(f"   LR Reduction: {'✓' if self.use_reduce_lr else '✗'}")
        print("="*70 + "\n")


# ============================================================================
# Main Training Pipeline
# ============================================================================

class KaggleTrainer:
    """
    Complete training pipeline optimized for Kaggle
    """
    
    def __init__(self, config: KaggleTrainingConfig):
        self.config = config
        self.model = None
        self.history = None
        self.results = {}
        
        # Create subdirectories
        self.model_dir = os.path.join(config.output_dir, 'models')
        self.plots_dir = os.path.join(config.output_dir, 'plots')
        self.logs_dir = os.path.join(config.output_dir, 'logs')
        
        for d in [self.model_dir, self.plots_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)
    
    def load_data(self):
        """
        Load and preprocess BreaKHis dataset
        """
        print("\n" + "="*70)
        print("📥 LOADING BREAKHIS DATASET")
        print("="*70)
        
        loader = BreaKHisDataLoader(
            base_path=self.config.data_path,
            image_size=(self.config.image_size, self.config.image_size)
        )
        
        # Load dataset (returns dict with train/val/test splits)
        dataset = loader.load_dataset(
            magnification=self.config.magnification,
            binary=self.config.binary
        )
        
        # Load actual images from paths
        print("\n📂 Loading images from disk...")
        X_train = loader.load_images_from_paths(dataset['train']['paths'])
        X_val = loader.load_images_from_paths(dataset['val']['paths'])
        X_test = loader.load_images_from_paths(dataset['test']['paths'])
        
        y_train = dataset['train']['labels']
        y_val = dataset['val']['labels']
        y_test = dataset['test']['labels']
        
        print(f"✅ Dataset loaded:")
        print(f"   Total images: {len(X_train) + len(X_val) + len(X_test)}")
        print(f"   Image shape: {X_train[0].shape}")
        print(f"   Classes: {np.unique(y_train)}")
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Apply Gabor filtering if enabled
        if self.config.use_gabor:
            print(f"\n🔬 Applying Gabor filtering...")
            gabor = GaborFilter()
            X_train = np.array([gabor.apply_multiscale_filter(img) for img in X_train])
            X_val = np.array([gabor.apply_multiscale_filter(img) for img in X_val])
            X_test = np.array([gabor.apply_multiscale_filter(img) for img in X_test])
            print(f"   ✅ Gabor filtering applied")
        
        # Class distribution
        from collections import Counter
        train_dist = Counter(y_train)
        print(f"\n⚖️  Training Set Distribution:")
        for cls, count in train_dist.items():
            print(f"   Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
        # Apply SMOTE if enabled
        if self.config.use_smote:
            print(f"\n🔄 Applying SMOTE balancing...")
            balancer = DataBalancer()
            X_train, y_train = balancer.balance_dataset(X_train, y_train)
            
            train_dist_balanced = Counter(y_train)
            print(f"   After SMOTE:")
            for cls, count in train_dist_balanced.items():
                print(f"   Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
        # Convert to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        num_classes = 2 if self.config.binary else 8
        
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Store data
        self.data = {
            'X_train': X_train,
            'y_train': y_train_cat,
            'X_val': X_val,
            'y_val': y_val_cat,
            'X_test': X_test,
            'y_test': y_test_cat,
            'y_test_labels': y_test,
            'num_classes': num_classes
        }
        
        # Clean up
        local_vars = locals()

        cleanup_targets = ['X', 'y', 'metadata', 'X_temp', 'y_temp']
        for var_name in cleanup_targets:
             if var_name in local_vars:
                  del local_vars[var_name]
        import gc
        gc.collect()
        
        print("✅ Data preparation complete!\n")
        
    def build_model(self):
        """
        Build Enhanced CVFBJTL-BCD model
        """
        print("\n" + "="*70)
        print("🏗️  BUILDING ENHANCED CVFBJTL-BCD MODEL")
        print("="*70)
        
        fusion = EnhancedFeatureFusion(
            input_shape=(self.config.image_size, self.config.image_size, 3),
            include_vit=self.config.use_vit
        )
        
        self.model = fusion.build_fusion_model(
            num_classes=self.data['num_classes']
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Model summary
        print(f"\n📋 Model Architecture Summary:")
        self.model.summary(print_fn=lambda x: print(f"   {x}"))
        
    def setup_callbacks(self):
        """
        Setup training callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.model_dir, 'best_model.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        if self.config.use_early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Reduce learning rate
        if self.config.use_reduce_lr:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.logs_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_logger = keras.callbacks.CSVLogger(
            os.path.join(self.logs_dir, 'training_log.csv')
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train_model(self):
        """
        Train the model
        """
        print("\n" + "="*70)
        print("🎓 TRAINING MODEL")
        print("="*70)
        
        callbacks = self.setup_callbacks()
        
        # Data augmentation
        if self.config.use_augmentation:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )
            
            print("✅ Data augmentation enabled")
            
            self.history = self.model.fit(
                datagen.flow(self.data['X_train'], self.data['y_train'], 
                           batch_size=self.config.batch_size),
                validation_data=(self.data['X_val'], self.data['y_val']),
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                self.data['X_train'], self.data['y_train'],
                validation_data=(self.data['X_val'], self.data['y_val']),
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✅ Training complete!")
        
    def evaluate_model(self):
        """
        Evaluate model on test set
        """
        print("\n" + "="*70)
        print("📊 EVALUATING MODEL")
        print("="*70)
        
        # Load best model
        best_model_path = os.path.join(self.model_dir, 'best_model.h5')
        if os.path.exists(best_model_path):
            self.model.load_weights(best_model_path)
            print("✅ Loaded best model weights")
        
        # Evaluate on test set
        test_results = self.model.evaluate(
            self.data['X_test'], self.data['y_test'],
            batch_size=self.config.batch_size,
            verbose=1
        )
        
        # Get predictions
        y_pred_prob = self.model.predict(self.data['X_test'])
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = self.data['y_test_labels']
        
        # Calculate metrics
        from sklearn.metrics import (
            classification_report, confusion_matrix, 
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\n🎯 Test Set Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        
        # Compare with paper
        paper_accuracy = 98.18
        improvement = accuracy * 100 - paper_accuracy
        
        print(f"\n📈 Comparison with Paper:")
        print(f"   Paper (CVFBJTL-BCD): {paper_accuracy}%")
        print(f"   Our Implementation:  {accuracy*100:.2f}%")
        print(f"   Improvement:         {improvement:+.2f}%")
        
        if accuracy * 100 > paper_accuracy:
            print(f"   🎉 BETTER THAN PAPER! ✨")
        elif accuracy * 100 > paper_accuracy - 0.5:
            print(f"   ✅ Comparable to paper!")
        else:
            print(f"   ⚠️  Below paper performance. Consider:")
            print(f"       - Training for more epochs")
            print(f"       - Enabling HHOA optimization")
            print(f"       - Using larger image size (224x224)")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(y_true, y_pred, 
                                      target_names=['Benign', 'Malignant'])
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        self.results = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'paper_accuracy': paper_accuracy,
            'improvement': float(improvement)
        }
        
        # Save results
        results_path = os.path.join(self.config.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\n✅ Results saved to: {results_path}")
        
    def generate_plots(self):
        """
        Generate visualization plots
        """
        print("\n" + "="*70)
        print("📊 GENERATING PLOTS")
        print("="*70)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        print("✅ Saved: training_history.png")
        
        # 2. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(self.results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print("✅ Saved: confusion_matrix.png")
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve, auc
        
        y_pred_prob = self.model.predict(self.data['X_test'])
        fpr, tpr, _ = roc_curve(self.data['y_test_labels'], y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        print("✅ Saved: roc_curve.png")
        
        plt.close('all')
        print("\n✅ All plots generated!")
        
    def generate_gradcam(self, num_samples: int = 10):
        """
        Generate Grad-CAM explanations
        """
        print("\n" + "="*70)
        print("🔍 GENERATING GRAD-CAM EXPLANATIONS")
        print("="*70)
        
        try:
            gradcam = GradCAM(self.model)
            
            # Select random samples
            indices = np.random.choice(len(self.data['X_test']), 
                                     size=min(num_samples, len(self.data['X_test'])),
                                     replace=False)
            
            gradcam_dir = os.path.join(self.plots_dir, 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)
            
            for i, idx in enumerate(indices):
                img = self.data['X_test'][idx]
                true_label = self.data['y_test_labels'][idx]
                
                # Generate Grad-CAM
                heatmap = gradcam.generate_heatmap(
                    img, 
                    class_idx=true_label,
                    layer_name='densenet_gap'
                )
                
                # Visualize
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img)
                axes[0].set_title('Original Image', fontsize=12)
                axes[0].axis('off')
                
                # Grad-CAM heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
                axes[1].axis('off')
                
                # Overlay
                overlay = img * 0.5 + heatmap[:, :, np.newaxis] * 0.5
                axes[2].imshow(overlay)
                axes[2].set_title('Overlay', fontsize=12)
                axes[2].axis('off')
                
                label_name = 'Benign' if true_label == 0 else 'Malignant'
                fig.suptitle(f'Sample {i+1} - True Label: {label_name}', 
                           fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(os.path.join(gradcam_dir, f'gradcam_{i+1}.png'), 
                          dpi=200, bbox_inches='tight')
                plt.close()
            
            print(f"✅ Generated {len(indices)} Grad-CAM visualizations")
            
        except Exception as e:
            print(f"⚠️  Grad-CAM generation failed: {e}")
            print("   (This is optional - training results are still valid)")
    
    def save_final_report(self):
        """
        Save comprehensive training report
        """
        print("\n" + "="*70)
        print("📝 GENERATING FINAL REPORT")
        print("="*70)
        
        report = []
        report.append("="*70)
        report.append("ENHANCED CVFBJTL-BCD TRAINING REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
        
        report.append("\n" + "="*70)
        report.append("CONFIGURATION")
        report.append("="*70)
        report.append(f"Image Size: {self.config.image_size}x{self.config.image_size}")
        report.append(f"Magnification: {self.config.magnification}")
        report.append(f"Epochs: {self.config.epochs}")
        report.append(f"Batch Size: {self.config.batch_size}")
        report.append(f"Learning Rate: {self.config.learning_rate}")
        
        report.append("\n" + "="*70)
        report.append("RESULTS")
        report.append("="*70)
        report.append(f"Test Accuracy:  {self.results['test_accuracy']*100:.2f}%")
        report.append(f"Test Precision: {self.results['test_precision']*100:.2f}%")
        report.append(f"Test Recall:    {self.results['test_recall']*100:.2f}%")
        report.append(f"Test F1-Score:  {self.results['test_f1']*100:.2f}%")
        
        report.append("\n" + "="*70)
        report.append("COMPARISON WITH PAPER")
        report.append("="*70)
        report.append(f"Paper Accuracy: {self.results['paper_accuracy']}%")
        report.append(f"Our Accuracy:   {self.results['test_accuracy']*100:.2f}%")
        report.append(f"Improvement:    {self.results['improvement']:+.2f}%")
        
        if self.results['improvement'] > 0:
            report.append("\n✨ ACHIEVEMENT: Exceeded paper performance! ✨")
        
        report.append("\n" + "="*70)
        report.append("CLASSIFICATION REPORT")
        report.append("="*70)
        report.append(self.results['classification_report'])
        
        report.append("\n" + "="*70)
        report.append("FILES GENERATED")
        report.append("="*70)
        report.append("Models:")
        report.append("  - best_model.h5")
        report.append("\nPlots:")
        report.append("  - training_history.png")
        report.append("  - confusion_matrix.png")
        report.append("  - roc_curve.png")
        report.append("  - gradcam/*.png (if generated)")
        report.append("\nData:")
        report.append("  - results.json")
        report.append("  - training_log.csv")
        
        report.append("\n" + "="*70)
        
        # Save report
        report_text = '\n'.join(report)
        report_path = os.path.join(self.config.output_dir, 'TRAINING_REPORT.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ Report saved to: {report_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main training pipeline
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Enhanced CVFBJTL-BCD Training on Kaggle".center(68) + "║")
    print("║" + "  Breast Cancer Diagnosis using Deep Learning".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Setup GPU
    gpu_available = setup_kaggle_gpu()
    if not gpu_available and IS_KAGGLE:
        print("\n⚠️  WARNING: No GPU detected on Kaggle!")
        print("   Please enable GPU in notebook settings.")
        print("   Settings → Accelerator → GPU (T4 x2)")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    optimize_tensorflow_for_kaggle()
    
    # Initialize configuration
    config = KaggleTrainingConfig()
    config.set_seeds()
    config.print_config()
    
    # Initialize trainer
    trainer = KaggleTrainer(config)
    
    # Training pipeline
    try:
        # 1. Load data
        trainer.load_data()
        
        # 2. Build model
        trainer.build_model()
        
        # 3. Train model
        start_time = time.time()
        trainer.train_model()
        training_time = time.time() - start_time
        print(f"\n⏱️  Total training time: {training_time/60:.1f} minutes")
        
        # 4. Evaluate model
        trainer.evaluate_model()
        
        # 5. Generate plots
        trainer.generate_plots()
        
        # 6. Generate Grad-CAM (optional)
        try:
            trainer.generate_gradcam(num_samples=10)
        except:
            print("⚠️  Skipping Grad-CAM (optional feature)")
        
        # 7. Save final report
        trainer.save_final_report()
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print(f"\n📁 All outputs saved to: {config.output_dir}")
        print("\n📊 Key Files:")
        print(f"   - Model: {os.path.join(trainer.model_dir, 'best_model.h5')}")
        print(f"   - Results: {os.path.join(config.output_dir, 'results.json')}")
        print(f"   - Report: {os.path.join(config.output_dir, 'TRAINING_REPORT.txt')}")
        print("\n✅ You can now download these files from Kaggle output!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import time
    main()
