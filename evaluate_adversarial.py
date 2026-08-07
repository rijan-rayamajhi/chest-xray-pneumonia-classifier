import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.config import Config
from src.dataset import DatasetManager
from src.adversarial import AdversarialEvaluator

def main():
    model_name = "DenseNet121"
    checkpoint_path = os.path.join(Config.MODEL_DIR, f"best_{model_name}.keras")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'. Run run_experiments.py first.")
        return
        
    print(f"Loading {model_name} model from {checkpoint_path}...")
    model = keras.models.load_model(checkpoint_path)
    
    dm = DatasetManager()
    _, _, test_fps, test_lbls = dm.load_all_data()
    
    evaluator = AdversarialEvaluator(model, model_name=model_name)
    epsilons = [0.0, 0.005, 0.01, 0.02, 0.05]
    robustness_results = evaluator.evaluate_sample_robustness(test_fps, test_lbls, epsilons=epsilons, num_samples=30)
    
    eps_vals = [r["epsilon"] for r in robustness_results]
    acc_vals = [r["accuracy"] * 100 for r in robustness_results]
    sens_vals = [r["sensitivity"] * 100 for r in robustness_results]
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(eps_vals, acc_vals, 'o-', linewidth=2, label="Accuracy (%)", color="#1f77b4")
    plt.plot(eps_vals, sens_vals, 's--', linewidth=2, label="Sensitivity (%)", color="#ff7f0e")
    plt.title(f"Adversarial FGSM Robustness Degradation ({model_name})", fontsize=11, fontweight="bold")
    plt.xlabel(r"FGSM Perturbation Magnitude ($\epsilon$)", fontsize=10)
    plt.ylabel("Performance Metric (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    
    curve_save_path = "outputs/figures/fgsm_robustness_degradation.png"
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig(curve_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved robustness degradation curve to {curve_save_path}")
    
    sample_idx = 0
    sample_fp = test_fps[sample_idx]
    sample_lbl = test_lbls[sample_idx]
    
    raw_bytes = tf.io.read_file(sample_fp)
    raw_img = tf.image.decode_jpeg(raw_bytes, channels=3)
    raw_resized = tf.image.resize(raw_img, Config.IMG_SIZE).numpy() / 255.0
    
    evaluator.plot_adversarial_attack_grid(
        sample_img=raw_resized,
        sample_lbl=sample_lbl,
        epsilons=[0.0, 0.01, 0.03],
        save_path="outputs/figures/fgsm_gradcam_shift.png"
    )
    print("Adversarial robustness evaluation completed.")

if __name__ == "__main__":
    main()
