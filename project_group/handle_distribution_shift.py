# =============================
#  Imports & Model Loading
# =============================
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, f1_score, accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from xgboost import XGBClassifier
matplotlib.use("TkAgg")

# Load original best model
best_model = joblib.load("best_xgboost_model.pkl")
print("\n Loaded best_xgboost_model.pkl")

# Load selected features
with open("selected_features.json", "r") as f:
    selected_features = json.load(f)

# Load datasets
X_train = pd.read_csv("datasets/X_train.csv")
y_train = pd.read_csv("datasets/y_train.csv").squeeze()
X_test2 = pd.read_csv("datasets/X_test_2.csv")
y_test2 = pd.read_csv("datasets/y_test_2_reduced.csv").squeeze()
X_test2_labeled = X_test2.iloc[:202]

X_train_sel = X_train[selected_features]
X_test2_sel = X_test2[selected_features]
X_test2_labeled_sel = X_test2_labeled[selected_features]

# Ensure consistent column types and order
X_train_sel.columns = X_train_sel.columns.astype(str)
X_test2_labeled_sel.columns = X_test2_labeled_sel.columns.astype(str)
X_test2_labeled_sel = X_test2_labeled_sel[X_train_sel.columns]

# =============================
#  Evaluation Function
# =============================
def evaluate_model(name, model, X_val, y_val, y_pred, y_proba):
    fixed_classes = np.arange(28)
    log_loss_val = log_loss(y_val, y_proba, labels=fixed_classes)

    y_ohe = label_binarize(y_val, classes=fixed_classes)
    def weighted_log_loss(y_true, y_pred):
        class_counts = np.sum(y_true, axis=0)
        class_counts_safe = np.where(class_counts == 0, 1e-8, class_counts)
        class_weights = 1.0 / class_counts_safe
        class_weights /= np.sum(class_weights)
        sample_weights = np.sum(y_true * class_weights, axis=1)
        return -np.mean(sample_weights * np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    acc = accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    wll = weighted_log_loss(y_ohe, y_proba)

    print(f"\n Evaluation - {name}")
    print(f"{'Accuracy:':<25}{acc:.4f}")
    print(f"{'F1 Macro:':<25}{f1_mac:.4f}")
    print(f"{'F1 Weighted:':<25}{f1_wt:.4f}")
    print(f"{'Log Loss:':<25}{log_loss_val:.4f}")
    print(f"{'Weighted Log Loss:':<25}{wll:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()

# =============================
#  Covariate Shift Detection
# =============================
"""
 Covariate Shift Summary:

Definition:
- Covariate shift occurs when the marginal distribution of features P(x) changes, while the conditional distribution P(y|x) remains the same.

Manifestation:
- The model's performance on the test data drops because the learned feature distribution during training no longer matches the distribution at inference time.

Detection Method:
- A domain classifier is trained to distinguish between training and test samples:
  - Training samples are labeled as 0, and test samples as 1, forming a binary classification task.
  - If the classifier achieves an AUC significantly higher than 0.5, this indicates a notable difference in feature distributions.
  - In our case, AUC = 0.7417, confirming a significant covariate shift.

Mitigation Strategy implemented in this project:
- Use the domain classifier to estimate the probability that each training sample could be misclassified as a test sample — this reflects its relative importance under the test distribution.
- Combine this importance weight with class weights (based on label distribution in the training set).
- Multiply the two to obtain a final sample weight for each training instance.
- Retrain the XGBoost model with `sample_weight` to create a shift-aware model that better generalizes to the shifted test distribution.
"""

print("\n Detecting Covariate Shift...")
X_domain = pd.concat([X_train_sel, X_test2_sel], axis=0)
y_domain = np.array([0] * len(X_train_sel) + [1] * len(X_test2_sel))

domain_clf = RandomForestClassifier(n_estimators=100, random_state=42)
domain_auc = cross_val_score(domain_clf, X_domain, y_domain, cv=5, scoring='roc_auc')
domain_auc_score = np.mean(domain_auc)
print(f" Domain Classifier AUC: {domain_auc_score:.4f}")

# =============================
#  Domain-Aware Model Correction if Needed
# =============================
if domain_auc_score >= 0.70:
    print(" Covariate shift detected. Applying correction...")
    domain_clf.fit(X_domain, y_domain)
    p_train_in_test = domain_clf.predict_proba(X_train_sel)[:, 1]

    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights_array))
    sample_weights_cs = y_train.map(class_weights_dict)
    sample_weights_combined = sample_weights_cs * p_train_in_test

    model_shift_aware = XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss', random_state=42)
    model_shift_aware.fit(X_train_sel, y_train, sample_weight=sample_weights_combined)
    joblib.dump(model_shift_aware, "model_shift_aware.pkl")
    print(" Trained and saved shift-aware model to model_shift_aware.pkl")
else:
    print(" No significant shift. Using original model.")
    model_shift_aware = best_model

# =============================
#  Label Shift Detection
# =============================
def compare_label_distributions(y_train, y_test):
    """
    Compare label distributions between the training set and the first 202 labeled samples in Test Set 2.

    Args:
        y_train: Series or 1D array, labels from the training set.
        y_test: Series or 1D array, labels from the first 202 samples of the test set.

    Returns:
        DataFrame: A table comparing the normalized label distributions in train vs. test (by class).
    """
    # Calculate label proportions (normalized frequency)
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()

    # Combine into a DataFrame for easy comparison
    df = pd.DataFrame({
        'Train': train_dist,
        'Test': test_dist
    }).fillna(0)  # Fill missing classes with 0 where needed

    # Plot the label distribution comparison
    ax = df.plot(kind='bar', figsize=(14, 5), title='Label Distribution Comparison (Train vs Test2-202)', rot=0)
    plt.ylabel("Proportion")
    plt.xlabel("Class Label")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("label_distribution_comparison.png")  # Save as PNG
    plt.close()  # Close the figure to avoid GUI popup

    return df


# Execute comparison
label_distribution_df = compare_label_distributions(y_train, y_test2)

# Print the top 10 classes by index (not frequency)
print("\nLabel distribution comparison (Top 10 classes by label index):")
print(label_distribution_df.head(10))

# =============================
#  Label Shift Detection
# =============================

def compare_label_distributions(y_train, y_test):
    """
    Compare label distributions between the training set and the first 202 labeled samples in Test Set 2.

    Args:
        y_train: Series or 1D array, labels from the training set.
        y_test: Series or 1D array, labels from the first 202 samples of the test set.

    Returns:
        DataFrame: A table comparing the normalized label distributions in train vs. test (by class).
    """
    # Calculate label proportions (normalized frequency)
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()

    # Combine into a DataFrame for easy comparison
    df = pd.DataFrame({
        'Train': train_dist,
        'Test': test_dist
    }).fillna(0)  # Fill missing classes with 0 where needed

    # Plot the label distribution comparison
    ax = df.plot(kind='bar', figsize=(14, 5), title='Label Distribution Comparison (Train vs Test2-202)', rot=0)
    plt.ylabel("Proportion")
    plt.xlabel("Class Label")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("label_distribution_comparison.png")  # Save as PNG
    plt.close()  # Close the figure to avoid GUI popup

    return df

# Execute comparison
label_distribution_df = compare_label_distributions(y_train, y_test2)

# Print the top 10 classes by index (not frequency)
print("\nLabel distribution comparison (Top 10 classes by label index):")
print(label_distribution_df.head(10))

# =============================
#  Label Shift Correction (Prior Probability Adjustment)
# =============================

def correct_label_shift(y_train, y_test, y_proba, epsilon=1e-6):
    """
    Apply prior probability correction to predicted probabilities to handle label shift.

    Args:
        y_train: array-like, training set labels
        y_test: array-like, test set labels (202 labeled)
        y_proba: numpy array, shape (n_samples, n_classes), predicted probabilities
        epsilon: small constant to avoid division by zero

    Returns:
        corrected_proba: numpy array, corrected probabilities
    """
    num_classes = y_proba.shape[1]
    train_prior = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_prior = pd.Series(y_test).value_counts(normalize=True).sort_index()

    # Ensure all classes are included
    for i in range(num_classes):
        if i not in train_prior:
            train_prior[i] = epsilon
        if i not in test_prior:
            test_prior[i] = epsilon

    # Align and normalize priors
    train_prior = train_prior.sort_index().values
    test_prior = test_prior.sort_index().values

    # Correct each predicted probability
    correction_factors = test_prior / (train_prior + epsilon)
    corrected_proba = y_proba * correction_factors
    corrected_proba /= corrected_proba.sum(axis=1, keepdims=True)

    return corrected_proba
y_proba_shift = model_shift_aware.predict_proba(X_test2_labeled_sel)
y_pred_shift = model_shift_aware.predict(X_test2_labeled_sel)
# Apply label shift correction and evaluate performance
corrected_proba = correct_label_shift(y_train, y_test2, y_proba_shift)
evaluate_model("Shift-Aware Model (Label-Shift Corrected)", model_shift_aware, X_test2_labeled_sel, y_test2, y_pred_shift, corrected_proba)

# =============================
#  Optional: Analyze Concept Drift Symptoms
# =============================
# Idea: Print confusion matrix where most misclassified classes are
# and check if they correspond to frequent semantic confusion
from collections import Counter

def analyze_concept_drift(y_true, y_pred):
    print("\n Top 10 Confused Class Pairs:")
    conf_matrix = confusion_matrix(y_true, y_pred)
    n = conf_matrix.shape[0]
    confusion_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and conf_matrix[i][j] > 0:
                confusion_pairs.append(((i, j), conf_matrix[i][j]))
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x[1], reverse=True)[:10]
    for (i, j), count in confusion_pairs:
        print(f"Class {i} → {j}: {count} samples")

analyze_concept_drift(y_test2, y_pred_shift)


# =============================
#  Evaluate Final & Best Models
# =============================
y_pred_shift = model_shift_aware.predict(X_test2_labeled_sel)
y_proba_shift = model_shift_aware.predict_proba(X_test2_labeled_sel)
evaluate_model("Final Model on Test2-202", model_shift_aware, X_test2_labeled_sel, y_test2, y_pred_shift, y_proba_shift)

# Use same alignment for best_model
X_test2_labeled_sel.columns = X_train_sel.columns
X_test2_labeled_sel = X_test2_labeled_sel[X_train_sel.columns]

y_pred_orig = best_model.predict(X_test2_labeled_sel)
y_proba_orig = best_model.predict_proba(X_test2_labeled_sel)
evaluate_model("Original Best Model on Test2-202", best_model, X_test2_labeled_sel, y_test2, y_pred_orig, y_proba_orig)


# =============================
#  Predict on Test Set 2 (1818 unlabeled samples) and Save preds_2.npy
# =============================
"""
Make predictions on the unlabeled portion of Test2 (1818 samples).
The model outputs a probability distribution over 28 classes for each sample.
This is saved in the required .npy format for submission.

Prediction shape is validated to ensure submission compatibility: [1818, 28]
"""

# Reload the full test set to ensure slicing is correct
X_test2_full = pd.read_csv("datasets/X_test_2.csv")
X_test2_unlabeled = X_test2_full.iloc[202:]
X_test2_unlabeled_sel = X_test2_unlabeled[selected_features]
X_test2_unlabeled_sel.columns = X_train_sel.columns  # Ensure column alignment

# Predict raw probabilities
raw_preds_2 = model_shift_aware.predict_proba(X_test2_unlabeled_sel)

# Apply label shift correction
corrected_preds_2 = correct_label_shift(y_train, y_test2, raw_preds_2)

# Save to .npy
np.save("preds_2.npy", corrected_preds_2)
print(f"preds_2.npy saved. Shape: {corrected_preds_2.shape}")

# Verify shape
assert corrected_preds_2.shape == (1818, 28), f"Shape mismatch: got {corrected_preds_2.shape}, expected (1818, 28)"
print(" Shape verified: preds_2.npy is ready for submission.")