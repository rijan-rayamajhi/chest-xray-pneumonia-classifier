import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report
)

from .config import Config

class ClinicalEvaluator:
    @staticmethod
    def compute_all_metrics(y_true, y_pred_prob, threshold=0.5):
        """Compute full suite of clinical classification metrics."""
        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        # Confusion matrix elements: TN, FP, FN, TP
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        # Clinical metrics
        acc = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for Pneumonia
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall for Normal
        precision = precision_score(y_true, y_pred, zero_division=0)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            roc_auc = 0.5
            
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(rec_curve, prec_curve)
        
        return {
            "Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "NPV": npv,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }

    @classmethod
    def compute_bootstrapped_ci(cls, y_true, y_pred_prob, n_bootstraps=1000, ci=95, seed=Config.SEED):
        """Compute non-parametric bootstrapped 95% Confidence Intervals."""
        np.random.seed(seed)
        n_samples = len(y_true)
        bootstrapped_metrics = []
        
        for _ in range(n_bootstraps):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue  # Skip if bootstrap sample has only 1 class
            metrics = cls.compute_all_metrics(y_true[indices], y_pred_prob[indices])
            bootstrapped_metrics.append(metrics)
            
        df_boot = pd.DataFrame(bootstrapped_metrics)
        lower_p = (100 - ci) / 2.0
        upper_p = 100 - lower_p
        
        summary = {}
        point_estimates = cls.compute_all_metrics(y_true, y_pred_prob)
        
        for key in ["Accuracy", "Sensitivity", "Specificity", "Precision", "ROC_AUC", "F1_Weighted"]:
            mean_val = point_estimates[key]
            lower_val = np.percentile(df_boot[key], lower_p)
            upper_val = np.percentile(df_boot[key], upper_p)
            summary[key] = f"{mean_val:.4f} ({lower_val:.4f} - {upper_val:.4f})"
            summary[f"{key}_raw"] = mean_val
            summary[f"{key}_CI_low"] = lower_val
            summary[f"{key}_CI_high"] = upper_val
            
        return summary

    @staticmethod
    def generate_latex_table(results_df):
        """Convert metrics comparison dataframe into publication-grade LaTeX snippet."""
        latex_str = results_df.to_latex(index=False, float_format="%.4f")
        return latex_str
