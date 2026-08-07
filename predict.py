import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.config import Config
from src.models import get_preprocess_fn
from src.explainability import GradCAMVisualizer

def predict_single_image(image_path, model_name="DenseNet121", output_path="outputs/figures/single_prediction_gradcam.png"):
    if not os.path.exists(image_path):
        print(f"Error: image file not found at {image_path}")
        sys.exit(1)
        
    checkpoint_path = os.path.join(Config.MODEL_DIR, f"best_{model_name}.keras")
    if not os.path.exists(checkpoint_path):
        print(f"Error: model checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    model = keras.models.load_model(checkpoint_path)
    
    raw_img_bytes = tf.io.read_file(image_path)
    raw_img = tf.image.decode_jpeg(raw_img_bytes, channels=3)
    raw_img_resized = tf.image.resize(raw_img, Config.IMG_SIZE).numpy() / 255.0
    
    img_array = np.expand_dims(raw_img_resized, axis=0)
    preprocess_fn = get_preprocess_fn(model_name)
    if preprocess_fn is not None:
        img_array = preprocess_fn(img_array * 255.0)
        
    pred_prob = float(model.predict(img_array, verbose=0)[0][0])
    pred_label = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"
    confidence = pred_prob if pred_prob >= 0.5 else (1.0 - pred_prob)
    
    print(f"File: {os.path.basename(image_path)} | Diagnosis: {pred_label} | Score: {pred_prob:.4f} ({confidence * 100:.2f}%)")
    
    visualizer = GradCAMVisualizer(model, model_name)
    heatmap = visualizer.make_gradcam_heatmap(img_array)
    overlay = visualizer.overlay_gradcam(raw_img_resized, heatmap, alpha=0.45)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img_resized)
    plt.title(f"Input: {os.path.basename(image_path)}")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM ({pred_label}, {confidence*100:.1f}%)")
    plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to input X-ray image")
    parser.add_argument("--model", type=str, default="DenseNet121", help="Model name")
    parser.add_argument("--output", type=str, default="outputs/figures/single_prediction_gradcam.png", help="Output figure path")
    args = parser.parse_args()
    
    if args.image is None:
        test_dir = os.path.join(Config.DATA_DIR, "test")
        sample_images = []
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    sample_images.append(os.path.join(root, f))
        if sample_images:
            args.image = np.random.choice(sample_images)
        else:
            print("Error: No test images found.")
            sys.exit(1)
            
    predict_single_image(args.image, args.model, args.output)
