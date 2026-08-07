import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras

from .config import Config
from .models import get_target_conv_layer

class GradCAMVisualizer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.target_layer_name = get_target_conv_layer(model_name, model)
        
    def make_gradcam_heatmap(self, img_array):
        """Generate Grad-CAM activation heatmap for a single image array (shape: 1, H, W, C)."""
        # Search for target layer in main model or nested backbone
        target_layer = None
        grad_model = None
        
        try:
            target_layer = self.model.get_layer(self.target_layer_name)
            grad_model = keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception:
            # Fallback: search within nested pretrained base model
            for layer in self.model.layers:
                if isinstance(layer, keras.Model):
                    try:
                        target_layer = layer.get_layer(self.target_layer_name)
                        grad_model = keras.Model(
                            inputs=layer.inputs,
                            outputs=[target_layer.output, layer.output]
                        )
                        break
                    except Exception:
                        continue

        if grad_model is None:
            # If target layer cannot be extracted, return dummy blank heatmap
            return np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH))

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            tape.watch(inputs)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, 0]

        # Compute gradient of output logit with respect to feature maps
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to keep positive influence only and normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def overlay_gradcam(self, original_img, heatmap, alpha=0.4):
        """Overlay JET colormap heatmap onto original image tensor."""
        # Rescale heatmap to 0-255
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis], (original_img.shape[0], original_img.shape[1])
        ).numpy().squeeze()
        
        # Colorize heatmap using jet colormap
        jet = plt.get_cmap("jet")
        jet_colors = jet(heatmap_resized)[:, :, :3]  # Drop alpha channel
        
        # Ensure original image is in range [0, 1]
        if original_img.max() > 1.0:
            norm_img = original_img / 255.0
        else:
            norm_img = original_img.copy()
            
        if norm_img.ndim == 2 or norm_img.shape[-1] == 1:
            norm_img = np.stack([norm_img.squeeze()] * 3, axis=-1)
            
        superimposed_img = jet_colors * alpha + norm_img * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 1)
        return superimposed_img

    def plot_explainability_grid(self, samples, save_path=None):
        """
        Plot grid of 4 diagnostic categories: True Positive, True Negative, False Positive, False Negative.
        samples: list of dicts with keys ['img', 'true_lbl', 'pred_prob', 'case_type']
        """
        fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
        if len(samples) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, sample in enumerate(samples):
            img = sample['img']
            true_lbl = Config.CLASS_NAMES[sample['true_lbl']]
            pred_prob = sample['pred_prob']
            pred_lbl = Config.CLASS_NAMES[int(pred_prob >= 0.5)]
            case_type = sample['case_type']
            
            img_tensor = np.expand_dims(img, axis=0)
            heatmap = self.make_gradcam_heatmap(img_tensor)
            overlay = self.overlay_gradcam(img, heatmap)
            
            # Subplot 1: Original X-Ray
            axes[i, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[i, 0].set_title(f"Case: {case_type}\nTrue: {true_lbl} | Pred: {pred_lbl} ({pred_prob:.2f})")
            axes[i, 0].axis("off")
            
            # Subplot 2: Heatmap
            axes[i, 1].imshow(heatmap, cmap="jet")
            axes[i, 1].set_title("Grad-CAM Heatmap")
            axes[i, 1].axis("off")
            
            # Subplot 3: Overlay
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Superimposed Focus Region")
            axes[i, 2].axis("off")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
