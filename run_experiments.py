import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

from src.config import Config
from src.dataset import DatasetManager
from src.trainer import ModelTrainer
from src.metrics import ClinicalEvaluator
from src.explainability import GradCAMVisualizer

def main():
    print("Starting Chest X-Ray Benchmark Experiments...")
    
    dm = DatasetManager()
    train_fps, train_lbls, test_fps, test_lbls = dm.load_all_data()
    
    print(f"Train samples: {len(train_fps)}, Test samples: {len(test_fps)}")
    
    tr_fps, val_fps, tr_lbls, val_lbls = train_test_split(
        train_fps, train_lbls, test_size=0.2, stratify=train_lbls, random_state=Config.SEED
    )
    
    class_weights = dm.compute_class_weights(tr_lbls)
    
    models_to_run = ["BaselineCNN", "DenseNet121", "ResNet50V2", "EfficientNetB0", "MobileNetV2"]
    benchmark_results = []
    roc_data = {}
    pr_data = {}
    trained_models = {}

    for model_name in models_to_run:
        print(f"\n--- Running: {model_name} ---")
        trainer = ModelTrainer(model_name)
        preprocess_fn = trainer.preprocess_fn
        
        tr_ds = dm.create_tf_dataset(tr_fps, tr_lbls, shuffle=True, augment=True, preprocess_fn=preprocess_fn)
        val_ds = dm.create_tf_dataset(val_fps, val_lbls, shuffle=False, augment=False, preprocess_fn=preprocess_fn)
        test_ds = dm.create_tf_dataset(test_fps, test_lbls, shuffle=False, augment=False, preprocess_fn=preprocess_fn)
        
        epochs = 10 if model_name != "BaselineCNN" else 12
        model, _, _ = trainer.train_model(
            tr_ds, val_ds, class_weights=class_weights, epochs=epochs, fine_tune_epochs=4 if model_name != "BaselineCNN" else 0
        )
        
        y_true, y_pred_prob = trainer.evaluate_predictions(model, test_ds)
        
        metrics_summary = ClinicalEvaluator.compute_bootstrapped_ci(y_true, y_pred_prob, n_bootstraps=500)
        metrics_summary["Model"] = model_name
        benchmark_results.append(metrics_summary)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
        roc_data[model_name] = (fpr, tpr, metrics_summary["ROC_AUC_raw"])
        pr_data[model_name] = (rec, prec, metrics_summary["F1_Weighted_raw"])
        trained_models[model_name] = (model, y_true, y_pred_prob, test_ds)
        
    df_res = pd.DataFrame(benchmark_results)
    cols = ["Model", "Accuracy", "Sensitivity", "Specificity", "Precision", "ROC_AUC", "F1_Weighted"]
    df_display = df_res[cols]
    
    print("\nBenchmark Results Summary:")
    print(df_display.to_string(index=False))
    
    csv_path = os.path.join(Config.TABLE_DIR, "benchmark_results.csv")
    tex_path = os.path.join(Config.TABLE_DIR, "benchmark_results.tex")
    df_res.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(ClinicalEvaluator.generate_latex_table(df_display))
        
    plt.figure(figsize=(12, 5))
    
    # ROC curve
    plt.subplot(1, 2, 1)
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5000)")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # PR curve
    plt.subplot(1, 2, 2)
    for model_name, (rec, prec, f1_score_val) in pr_data.items():
        plt.plot(rec, prec, label=f"{model_name} (F1 = {f1_score_val:.4f})", linewidth=2)
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_fig_path = os.path.join(Config.FIGURE_DIR, "roc_pr_curves_comparison.png")
    plt.savefig(curves_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    best_model_name = df_res.sort_values(by="ROC_AUC_raw", ascending=False).iloc[0]["Model"]
    print(f"\nGenerating Grad-CAM visualizations for top model: {best_model_name}")
    
    best_model, y_true, y_pred_prob, test_ds = trained_models[best_model_name]
    visualizer = GradCAMVisualizer(best_model, best_model_name)
    
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0][0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0][0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0][0] if len(np.where((y_true == 0) & (y_pred == 1))[0]) > 0 else tp_idx
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0][0] if len(np.where((y_true == 1) & (y_pred == 0))[0]) > 0 else tn_idx
    
    sample_indices = [
        ("True Positive (Pneumonia)", tp_idx),
        ("True Negative (Normal)", tn_idx),
        ("False Positive", fp_idx),
        ("False Negative", fn_idx)
    ]
    
    samples_for_viz = []
    for case_type, idx in sample_indices:
        fp = test_fps[idx]
        lbl = test_lbls[idx]
        prob = y_pred_prob[idx]
        
        img_bytes = tf.io.read_file(fp)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, Config.IMG_SIZE).numpy() / 255.0
        
        samples_for_viz.append({
            'img': img,
            'true_lbl': lbl,
            'pred_prob': prob,
            'case_type': case_type
        })
        
    gradcam_fig_path = os.path.join(Config.FIGURE_DIR, "gradcam_explainability_grid.png")
    visualizer.plot_explainability_grid(samples_for_viz, save_path=gradcam_fig_path)
    
    print("\nBenchmark completed successfully.")

if __name__ == "__main__":
    main()
