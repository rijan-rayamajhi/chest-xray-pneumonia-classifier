import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from .config import Config
from .explainability import GradCAMVisualizer

class AdversarialEvaluator:
    """
    Evaluates model robustness against Fast Gradient Sign Method (FGSM) adversarial perturbations.
    Formulation: X_adv = X + epsilon * sign(grad_X J(theta, X, y))
    """
    def __init__(self, model, model_name="DenseNet121"):
        self.model = model
        self.model_name = model_name
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def generate_fgsm_image(self, img_array, label, epsilon=0.01):
        """Generate FGSM perturbed image for a single 3D image array (H, W, C)."""
        if epsilon == 0.0:
            return img_array.copy()
            
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)
        lbl_tensor = tf.convert_to_tensor([[label]], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            pred = self.model(img_tensor, training=False)
            loss = self.loss_fn(lbl_tensor, pred)
            
        grad = tape.gradient(loss, img_tensor)
        signed_grad = tf.sign(grad)
        adv_tensor = img_tensor + epsilon * signed_grad
        adv_tensor = tf.clip_by_value(adv_tensor, 0.0, 1.0)
        return adv_tensor.numpy()[0]

    def evaluate_sample_robustness(self, test_fps, test_lbls, epsilons=[0.0, 0.005, 0.01, 0.02, 0.05], num_samples=30):
        """Evaluate Accuracy degradation across epsilon levels on a representative test subset."""
        print(f"Evaluating FGSM Adversarial Robustness on {num_samples} test samples for {self.model_name}...")
        results = []
        
        # Load sample raw images
        sample_imgs = []
        sample_labels = []
        for i in range(min(num_samples, len(test_fps))):
            raw_bytes = tf.io.read_file(test_fps[i])
            raw_img = tf.image.decode_jpeg(raw_bytes, channels=3)
            img_res = tf.image.resize(raw_img, Config.IMG_SIZE).numpy() / 255.0
            sample_imgs.append(img_res)
            sample_labels.append(test_lbls[i])
            
        sample_labels = np.array(sample_labels)
        
        for eps in epsilons:
            preds = []
            for img, lbl in zip(sample_imgs, sample_labels):
                adv_img = self.generate_fgsm_image(img, lbl, epsilon=eps)
                p = float(self.model.predict(np.expand_dims(adv_img, axis=0), verbose=0)[0][0])
                preds.append(p)
                
            preds = np.array(preds)
            y_pred = (preds >= 0.5).astype(int)
            acc = float(np.mean(sample_labels == y_pred))
            
            pneu_mask = (sample_labels == 1)
            sens = float(np.mean(y_pred[pneu_mask] == 1)) if np.sum(pneu_mask) > 0 else 0.0
            
            results.append({
                "epsilon": eps,
                "accuracy": acc,
                "sensitivity": sens
            })
            print(f"  Epsilon: {eps:.3f} | Accuracy: {acc*100:.1f}% | Sensitivity: {sens*100:.1f}%")
            
        return results

    def plot_adversarial_attack_grid(self, sample_img, sample_lbl, epsilons=[0.0, 0.01, 0.03], save_path="outputs/figures/fgsm_gradcam_shift.png"):
        """Plot image and Grad-CAM shift under increasing adversarial perturbation epsilon."""
        visualizer = GradCAMVisualizer(self.model, self.model_name)
        
        plt.figure(figsize=(10, 3.5 * len(epsilons)))
        
        for idx, eps in enumerate(epsilons):
            adv_img = self.generate_fgsm_image(sample_img, sample_lbl, epsilon=eps)
            adv_tensor = np.expand_dims(adv_img, axis=0)
            
            pred_prob = float(self.model.predict(adv_tensor, verbose=0)[0][0])
            pred_label = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"
            
            heatmap = visualizer.make_gradcam_heatmap(adv_tensor)
            overlay = visualizer.overlay_gradcam(adv_img, heatmap, alpha=0.45)
            
            plt.subplot(len(epsilons), 2, idx * 2 + 1)
            plt.imshow(adv_img)
            plt.title(fr"FGSM Perturbation ($\epsilon={eps:.2f}$)" + f"\nTrue: {Config.CLASS_NAMES[sample_lbl]} | Pred: {pred_label} ({pred_prob:.3f})", fontsize=10)
            plt.axis("off")
            
            plt.subplot(len(epsilons), 2, idx * 2 + 2)
            plt.imshow(overlay)
            plt.title(fr"Grad-CAM Shift ($\epsilon={eps:.2f}$)" + "\nAttention Map", fontsize=10)
            plt.axis("off")
            
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved adversarial robustness visualization to {save_path}")
