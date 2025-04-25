import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Global save path for shift detection results
SAVE_PATH = "shift_detection_result"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def weighted_log_loss(y_true, y_pred, epsilon=1e-15):
    """
    Compute the weighted log loss with class-based sample weights.

    Parameters:
    - y_true: (N, C) One-hot encoded true labels.
    - y_pred: (N, C) Predicted probabilities.
    - epsilon: Small value to avoid log(0).

    Returns:
    - loss (float): Weighted log loss.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 避免 log(0)
    
    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))

    return loss


def evaluate_model(name, model, _, y_val, y_pred, y_proba):
    """
    Evaluate model performance using accuracy, F1 scores, and weighted log loss.

    Args:
    - name (str): Model name for display and file saving.
    - model: Trained model object (not used directly but included for extensibility).
    - X_val (pd.DataFrame): Validation feature set (not used directly).
    - y_val (pd.Series): Validation labels.
    - y_pred (np.ndarray): Predicted labels.
    - y_proba (np.ndarray): Predicted probabilities.

    Returns:
    - dict: Evaluation metrics.
    """
    # Validate inputs
    if len(y_val) != len(y_pred) or len(y_val) != y_proba.shape[0]:
        raise ValueError("y_val, y_pred, and y_proba must have the same number of samples")
    
    # Get valid classes from y_val
    valid_classes = np.unique(y_val)
    if len(valid_classes) == 0:
        raise ValueError("y_val contains no valid classes")
    
    # Get number of classes from y_proba
    num_classes = y_proba.shape[1]
    
    # One-hot encode y_val for all possible classes
    y_ohe = label_binarize(y_val, classes=np.arange(num_classes))
    
    # Extract valid classes for weighted log loss
    y_ohe_valid = y_ohe[:, valid_classes]
    y_proba_valid = y_proba[:, valid_classes]
    
    # Compute metrics
    acc = accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    wll = weighted_log_loss(y_ohe_valid, y_proba_valid)

    # Print results
    print(f"\nEvaluation - {name}")
    print(f"{'Accuracy:':<25}{acc:.4f}")
    print(f"{'F1 Macro:':<25}{f1_mac:.4f}")
    print(f"{'F1 Weighted:':<25}{f1_wt:.4f}")
    print(f"{'Weighted Log Loss:':<25}{wll:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"conf_matrix_{name.replace(' ', '_').lower()}.png"))
    plt.close()

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "weighted_log_loss": wll
    }