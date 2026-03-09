# utils/metrics.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import numpy as np


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates various classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities (for AUC). Shape (n_samples, n_classes).

    Returns:
        dict: Dictionary containing metric names and values.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa_value = cohen_kappa_score(y_true, y_pred)
    metrics['kappa'] = float(kappa_value)

    # Calculate AUC if probabilities are provided and it's not binary classification only
    # --- 计算并添加每个类别的召回率 ---
    num_classes = len(np.unique(y_true))
    if num_classes >= 2 and len(y_true) > 0:  # Ensure there are at least 2 classes and data exists
        try:
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))  # Ensure consistent label order
            # per_class_recall = cm.diagonal() / cm.sum(axis=1) # Recall = TP / (TP + FN)
            # Handle division by zero if a class has no true instances (shouldn't happen with np.unique labels if len>1)
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress potential division by zero warnings
                per_class_recall = cm.diagonal() / cm.sum(axis=1)
                per_class_recall[np.isnan(per_class_recall)] = 0.0  # Set recall to 0 if denominator is 0

            unique_labels = np.unique(y_true)
            for i, label in enumerate(unique_labels):
                metric_name = f'recall_class_{label}'  # e.g., recall_class_0, recall_class_1
                # In binary case, recall_class_1 is Sensitivity, recall_class_0 is Specificity's complement (or TNR)
                # It might be clearer to name them sensitivity and specificity for binary
                if num_classes == 2:
                    if label == 1:  # Assuming 1 is the positive class
                        metric_name = 'sensitivity'  # Recall for positive class
                    elif label == 0:
                        # Recall for negative class (True Negative Rate or Specificity)
                        # Specificity = TN / (TN + FP). Need TN and FP from cm.
                        if cm.shape == (2, 2):
                            tn = cm[0, 0]
                            fp = cm[0, 1]
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            metrics['specificity'] = float(specificity)
                            # Keep recall_class_0 as well if desired, or just use specificity
                            # metrics[f'recall_class_{label}'] = float(per_class_recall[i])
                            continue  # Skip adding recall_class_0 if specificity is added

                metrics[metric_name] = float(per_class_recall[i])

        except Exception as e:
            print(f"Warning: Could not calculate per-class recall/specificity: {e}")
            # Optionally add NaN placeholders
            unique_labels_fallback = np.unique(y_true) if len(y_true) > 0 else [0, 1]  # Fallback labels
            for label in unique_labels_fallback:
                metrics[f'recall_class_{label}'] = np.nan
                if num_classes == 2 and label == 1: metrics['sensitivity'] = np.nan
                if num_classes == 2 and label == 0: metrics['specificity'] = np.nan

    if y_prob is not None and len(y_true) > 1 and len(np.unique(y_true)) > 1:
        try:
            if num_classes == 2:
                # 对于二分类，通常使用正类的概率
                # 确保 y_prob 有至少两列
                if y_prob.shape[1] >= 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    print("Warning: Cannot calculate AUC for binary case, y_prob shape is unexpected.")
                    metrics['auc'] = np.nan  # 或者 0.0 或 None
            elif num_classes > 2:
                # 对于多分类，使用 One-vs-Rest (OvR) 宏平均 AUC
                metrics['auc'] = float(roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr'))
                # 可以选择性添加 'auc_ovr_weighted'
                # metrics['auc_ovr_weighted'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
        except ValueError as e:
            print(f"Could not calculate AUC: {e}")
            metrics['auc'] = np.nan  # 遇到错误时设为 NaN
    else:
        # 如果不满足计算AUC的条件
        metrics['auc'] = np.nan

    # metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist() # Optional

    return metrics
