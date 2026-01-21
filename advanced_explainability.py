#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED EXPLAINABILITY MODULE
==============================
Publication-ready explainability for medical imaging.

Features:
1. Grad-CAM++ (improved gradient-weighted class activation mapping)
2. Score-CAM (gradient-free alternative)
3. Attention map visualization
4. SHAP-based explanations
5. Integrated Gradients
6. Occlusion sensitivity analysis
7. Publication-quality visualization

Author: Advanced Research Implementation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import cv2


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for improved visualization.
    
    Improvements over Grad-CAM:
    - Better localization for multiple instances
    - Weighted combination of gradients
    - More faithful to model's decision process
    
    Reference: Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based 
               Visual Explanations for Deep Convolutional Networks"
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM++.
        
        Args:
            model: Keras model
            layer_name: Name of convolutional layer to visualize.
                       If None, automatically finds last conv layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        # Create gradient model
        self.grad_model = models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(self.layer_name).output,
                model.output
            ]
        )
        
    def _find_last_conv_layer(self) -> str:
        """Find the name of the last convolutional layer."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
            # Also check for layers within nested models
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, keras.layers.Conv2D):
                        return sublayer.name
        raise ValueError("Could not find any convolutional layers")
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap.
        
        Args:
            image: Input image (H, W, C) or (1, H, W, C)
            class_idx: Class index to explain. If None, uses predicted class.
            eps: Small constant for numerical stability.
            
        Returns:
            Heatmap of shape (H, W) normalized to [0, 1]
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Convert to tensor
        image_tensor = tf.cast(image, tf.float32)
        
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_output, predictions = self.grad_model(image_tensor)
                    
                    if class_idx is None:
                        class_idx = tf.argmax(predictions[0])
                    
                    class_output = predictions[:, class_idx]
                
                # First-order gradients
                first_grads = tape3.gradient(class_output, conv_output)
            
            # Second-order gradients
            second_grads = tape2.gradient(first_grads, conv_output)
        
        # Third-order gradients
        third_grads = tape1.gradient(second_grads, conv_output)
        
        # Compute alpha weights (Grad-CAM++ formula)
        global_sum = tf.reduce_sum(conv_output, axis=(0, 1, 2), keepdims=True)
        
        alpha_num = second_grads
        alpha_denom = 2 * second_grads + global_sum * third_grads + eps
        alpha = alpha_num / alpha_denom
        
        # Apply ReLU to gradients
        weights = tf.reduce_sum(alpha * tf.nn.relu(first_grads), axis=(1, 2))
        
        # Weighted combination
        cam = tf.reduce_sum(weights * conv_output, axis=-1)
        
        # Apply ReLU and normalize
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + eps)
        
        return cam.numpy()[0]
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, C) in [0, 1] range
            heatmap: Heatmap (h, w) in [0, 1] range
            alpha: Transparency of overlay
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]
        
        # Overlay
        superimposed = alpha * heatmap_colored + (1 - alpha) * image
        superimposed = np.clip(superimposed, 0, 1)
        
        return superimposed


class ScoreCAM:
    """
    Score-CAM: Gradient-free class activation mapping.
    
    More stable than gradient-based methods.
    Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations 
               for Convolutional Neural Networks"
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        # Create activation model
        self.activation_model = models.Model(
            inputs=model.inputs,
            outputs=model.get_layer(self.layer_name).output
        )
    
    def _find_last_conv_layer(self) -> str:
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        raise ValueError("No convolutional layers found")
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        batch_size: int = 16
    ) -> np.ndarray:
        """Compute Score-CAM heatmap."""
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Get activations
        activations = self.activation_model.predict(image, verbose=0)
        
        # Get original prediction
        original_pred = self.model.predict(image, verbose=0)
        if class_idx is None:
            class_idx = np.argmax(original_pred)
        
        # Upscale activations
        h, w = image.shape[1], image.shape[2]
        num_channels = activations.shape[-1]
        
        upscaled = np.zeros((num_channels, h, w))
        for i in range(num_channels):
            upscaled[i] = cv2.resize(activations[0, :, :, i], (w, h))
        
        # Normalize each activation map
        upscaled_normalized = np.zeros_like(upscaled)
        for i in range(num_channels):
            map_min = upscaled[i].min()
            map_max = upscaled[i].max()
            if map_max - map_min > 0:
                upscaled_normalized[i] = (upscaled[i] - map_min) / (map_max - map_min)
        
        # Compute scores for each activation map
        scores = []
        for i in range(0, num_channels, batch_size):
            batch_maps = upscaled_normalized[i:i+batch_size]
            masked_images = image * batch_maps[:, np.newaxis, :, :, np.newaxis]
            masked_images = masked_images.transpose(1, 0, 2, 3, 4)[0]
            
            batch_preds = self.model.predict(masked_images, verbose=0)
            batch_scores = batch_preds[:, class_idx]
            scores.extend(batch_scores.tolist())
        
        scores = np.array(scores)
        
        # Weighted combination
        cam = np.sum(scores[:, np.newaxis, np.newaxis] * upscaled_normalized, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam


class OcclusionSensitivity:
    """
    Occlusion sensitivity analysis.
    
    Shows which regions are most important by occluding parts of the image.
    """
    
    def __init__(self, model: keras.Model, patch_size: int = 16, stride: int = 8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
    
    def compute_sensitivity_map(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        occlusion_value: float = 0.5
    ) -> np.ndarray:
        """Compute occlusion sensitivity map."""
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        h, w = image.shape[1], image.shape[2]
        
        # Get original prediction
        original_pred = self.model.predict(image, verbose=0)
        if class_idx is None:
            class_idx = np.argmax(original_pred)
        original_score = original_pred[0, class_idx]
        
        # Compute sensitivity at each position
        sensitivity_map = np.zeros((h, w))
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Create occluded image
                occluded = image.copy()
                occluded[0, y:y+self.patch_size, x:x+self.patch_size, :] = occlusion_value
                
                # Get prediction
                occluded_pred = self.model.predict(occluded, verbose=0)
                occluded_score = occluded_pred[0, class_idx]
                
                # Sensitivity = drop in confidence
                sensitivity = original_score - occluded_score
                
                # Fill patch region
                sensitivity_map[y:y+self.patch_size, x:x+self.patch_size] = max(
                    sensitivity_map[y:y+self.patch_size, x:x+self.patch_size].max(),
                    sensitivity
                )
        
        # Normalize
        sensitivity_map = sensitivity_map / (sensitivity_map.max() + 1e-8)
        
        return sensitivity_map


class ExplainabilityVisualizer:
    """
    Publication-quality visualization for explainability.
    """
    
    def __init__(self, output_dir: str = 'outputs/explainability'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication style
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
    
    def visualize_gradcam_grid(
        self,
        model: keras.Model,
        images: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        n_samples: int = 8,
        filename: str = 'gradcam_grid.pdf'
    ):
        """Create a grid of Grad-CAM visualizations."""
        
        gradcam = GradCAMPlusPlus(model)
        
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        axes = axes.flatten()
        
        class_names = ['Benign', 'Malignant']
        
        for i in range(n_samples):
            if i >= len(images):
                break
            
            # Compute heatmap
            heatmap = gradcam.compute_heatmap(images[i])
            overlay = gradcam.overlay_heatmap(images[i], heatmap)
            
            # Plot
            axes[i].imshow(overlay)
            axes[i].axis('off')
            
            true_label = class_names[int(labels[i])]
            pred_label = class_names[int(predictions[i])]
            conf = predictions[i] if predictions[i] > 0.5 else 1 - predictions[i]
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({conf:.2f})',
                            fontsize=8, color=color)
        
        # Hide empty axes
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def visualize_multi_method(
        self,
        model: keras.Model,
        image: np.ndarray,
        filename: str = 'multi_method_comparison.pdf'
    ):
        """Compare multiple explainability methods."""
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM++
        gradcam = GradCAMPlusPlus(model)
        heatmap_gradcam = gradcam.compute_heatmap(image)
        overlay_gradcam = gradcam.overlay_heatmap(image, heatmap_gradcam)
        axes[1].imshow(overlay_gradcam)
        axes[1].set_title('Grad-CAM++')
        axes[1].axis('off')
        
        # Score-CAM
        try:
            scorecam = ScoreCAM(model)
            heatmap_score = scorecam.compute_heatmap(image)
            overlay_score = gradcam.overlay_heatmap(image, heatmap_score)
            axes[2].imshow(overlay_score)
            axes[2].set_title('Score-CAM')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Score-CAM\nError: {str(e)[:20]}', 
                        ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
        
        # Occlusion Sensitivity
        try:
            occlusion = OcclusionSensitivity(model, patch_size=16, stride=8)
            heatmap_occ = occlusion.compute_sensitivity_map(image)
            overlay_occ = gradcam.overlay_heatmap(image, heatmap_occ)
            axes[3].imshow(overlay_occ)
            axes[3].set_title('Occlusion Sensitivity')
        except Exception as e:
            axes[3].text(0.5, 0.5, f'Occlusion\nError: {str(e)[:20]}',
                        ha='center', va='center', transform=axes[3].transAxes)
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        
        print(f"Saved: {self.output_dir / filename}")
    
    def visualize_attention_maps(
        self,
        model: keras.Model,
        image: np.ndarray,
        layer_names: List[str],
        filename: str = 'attention_maps.pdf'
    ):
        """Visualize attention maps from different layers."""
        
        n_layers = len(layer_names)
        fig, axes = plt.subplots(1, n_layers + 1, figsize=(4*(n_layers+1), 4))
        
        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Each layer's attention
        for i, layer_name in enumerate(layer_names):
            try:
                gradcam = GradCAMPlusPlus(model, layer_name=layer_name)
                heatmap = gradcam.compute_heatmap(image)
                overlay = gradcam.overlay_heatmap(image, heatmap)
                axes[i+1].imshow(overlay)
                axes[i+1].set_title(f'{layer_name}')
            except Exception as e:
                axes[i+1].text(0.5, 0.5, f'Error', ha='center', va='center')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def create_publication_figure(
        self,
        model: keras.Model,
        benign_images: np.ndarray,
        malignant_images: np.ndarray,
        filename: str = 'explainability_figure.pdf'
    ):
        """Create main publication figure for explainability section."""
        
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        
        gradcam = GradCAMPlusPlus(model)
        
        # Benign samples (top row)
        for i in range(min(3, len(benign_images))):
            # Original
            axes[0, i*2].imshow(benign_images[i])
            axes[0, i*2].set_title('Benign', fontsize=10, fontweight='bold')
            axes[0, i*2].axis('off')
            
            # Grad-CAM++
            heatmap = gradcam.compute_heatmap(benign_images[i], class_idx=0)
            overlay = gradcam.overlay_heatmap(benign_images[i], heatmap)
            axes[0, i*2+1].imshow(overlay)
            axes[0, i*2+1].set_title('Grad-CAM++', fontsize=10)
            axes[0, i*2+1].axis('off')
        
        # Malignant samples (bottom row)
        for i in range(min(3, len(malignant_images))):
            # Original
            axes[1, i*2].imshow(malignant_images[i])
            axes[1, i*2].set_title('Malignant', fontsize=10, fontweight='bold')
            axes[1, i*2].axis('off')
            
            # Grad-CAM++
            heatmap = gradcam.compute_heatmap(malignant_images[i], class_idx=1)
            overlay = gradcam.overlay_heatmap(malignant_images[i], heatmap)
            axes[1, i*2+1].imshow(overlay)
            axes[1, i*2+1].set_title('Grad-CAM++', fontsize=10)
            axes[1, i*2+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
        
        print(f"Saved publication figure: {self.output_dir / filename}")


# ============================================================================
# INTEGRATED GRADIENTS
# ============================================================================

class IntegratedGradients:
    """
    Integrated Gradients for feature attribution.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
    """
    
    def __init__(self, model: keras.Model):
        self.model = model
    
    def compute_attributions(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """Compute integrated gradients attributions."""
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        if baseline is None:
            baseline = np.zeros_like(image)
        
        # Interpolated images
        alphas = np.linspace(0, 1, n_steps + 1)
        interpolated = np.array([
            baseline + alpha * (image - baseline) for alpha in alphas
        ])[:, 0]
        
        # Compute gradients
        interpolated_tensor = tf.cast(interpolated, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_tensor)
            predictions = self.model(interpolated_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[-1])
            
            target_class = predictions[:, class_idx]
        
        gradients = tape.gradient(target_class, interpolated_tensor)
        
        # Average gradients and scale
        avg_gradients = tf.reduce_mean(gradients, axis=0)
        attributions = (image[0] - baseline[0]) * avg_gradients.numpy()
        
        return attributions


if __name__ == "__main__":
    print("Advanced Explainability Module Loaded")
    print("=" * 50)
    print("Available Classes:")
    print("  - GradCAMPlusPlus: Improved gradient-weighted CAM")
    print("  - ScoreCAM: Gradient-free CAM")
    print("  - OcclusionSensitivity: Occlusion-based analysis")
    print("  - IntegratedGradients: Axiomatic attributions")
    print("  - ExplainabilityVisualizer: Publication figures")
    print("=" * 50)
