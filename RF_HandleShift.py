import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, classification_report, make_scorer
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import ks_2samp
from utils import weighted_log_loss
import os
import re
from imblearn.over_sampling import SMOTE

# Global save path
SAVE_PATH = "rf_after_shift"
os.makedirs(SAVE_PATH, exist_ok=True)

# === Step 2: Domain Classifier for Shift Detection ===
def detect_shift_with_domain_classifier(X_train, X_test2_labeled):
    """
    Use a Random Forest classifier to detect distribution shift between training and test data.
    
    Parameters:
    - X_train: Training features (numpy array or DataFrame)
    - X_test2_labeled: Labeled test features (first 202 samples, numpy array or DataFrame)
    
    Returns:
    - auc_mean: Mean ROC AUC score
    - auc_std: Standard deviation of ROC AUC scores
    - feature_importances: Feature importances from the domain classifier
    - p_train_in_test: Probability of each training sample being classified as test sample
    """
    # Convert to DataFrame if numpy array
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test2_labeled, np.ndarray):
        X_test2_labeled = pd.DataFrame(X_test2_labeled)
    
    # Combine data and create domain labels (0: train, 1: test)
    X_domain = pd.concat([X_train, X_test2_labeled], axis=0)
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test2_labeled))
    
    # Standardize the data
    scaler = StandardScaler()
    X_domain = scaler.fit_transform(X_domain)
    X_domain = pd.DataFrame(X_domain)
    
    # 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in kf.split(X_domain, y_domain):
        X_tr, X_val = X_domain.iloc[train_idx], X_domain.iloc[val_idx]
        y_tr, y_val = y_domain[train_idx], y_domain[val_idx]
        
        clf_rf = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
        clf_rf.fit(X_tr, y_tr)
        y_val_prob = clf_rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_prob)
        auc_scores.append(auc)
    
    # Train on full data to get feature importances and probabilities
    clf_rf_full = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
    clf_rf_full.fit(X_domain, y_domain)
    feature_importances = clf_rf_full.feature_importances_
    p_train_in_test = clf_rf_full.predict_proba(X_train)[:, 1]
    
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    return auc_mean, auc_std, feature_importances, p_train_in_test

# === Step 3: Covariate Shift Detection with KS Test ===
def detect_shift_with_ks_test(X_train, X_test2_labeled):
    """
    Use KS test to detect covariate shift between training and test data.
    
    Parameters:
    - X_train: Training features (numpy array)
    - X_test2_labeled: Labeled test features (first 202 samples, numpy array)
    
    Returns:
    - ks_df: DataFrame with KS test p-values for each feature
    """
    ks_results = []
    for i in range(X_train.shape[1]):
        stat, p_value = ks_2samp(X_train[:, i], X_test2_labeled[:, i])
        ks_results.append((i, p_value))
    ks_df = pd.DataFrame(ks_results, columns=['feature', 'p_value']).sort_values(by='p_value')
    return ks_df

# === Step 4: Feature Selection ===
def select_features(X_train, X_test2, pairwise_pval_filepath, ks_df, top_k=250):
    """
    Select features by removing shifted features and intersecting with pairwise p-value ranking.
    
    Parameters:
    - X_train: Training features (numpy array)
    - X_test2: Test2 features (numpy array)
    - pairwise_pval_filepath: Path to pairwise p-value feature ranking file
    - ks_df: DataFrame with KS test p-values
    - top_k: Number of top features to select
    
    Returns:
    - X_train_selected: Selected training features
    - X_test2_labeled_selected: Selected labeled test features
    - X_test2_unlabeled_selected: Selected unlabeled test features
    - final_selected_feats: List of selected feature indices
    """
    # Load pairwise p-value feature ranking
    def load_feature_indices(filepath, column="feature_idx"):
        df = pd.read_csv(filepath)
        if column in df.columns:
            return df[column].astype(int).tolist()
        else:
            return df.iloc[:, 0].astype(int).tolist()
    
    pairwise_pval_feats = load_feature_indices(pairwise_pval_filepath)
    
    # Remove shifted features (p-value < 1e-3)
    shifted_feats = ks_df[ks_df['p_value'] < 1e-3]['feature'].tolist()
    non_shifted_feats = list(set(np.arange(X_train.shape[1])) - set(shifted_feats))
    
    # Select intersection with pairwise p-value ranking
    final_selected_feats = [f for f in pairwise_pval_feats if f in non_shifted_feats]
    final_selected_feats = final_selected_feats[:top_k]
    
    # Apply feature selection
    X_train_selected = X_train[:, final_selected_feats]
    X_test2_labeled_selected = X_test2[:202, final_selected_feats]
    X_test2_unlabeled_selected = X_test2[202:, final_selected_feats]
    
    return X_train_selected, X_test2_labeled_selected, X_test2_unlabeled_selected, final_selected_feats

# === Step 5: Mitigate Covariate Shift ===
def mitigate_covariate_shift(X_train, y_train, p_train_in_test, percentile=90):
    """
    Mitigate covariate shift by combining domain classifier weights with class weights for sample reweighting.
    
    Parameters:
    - X_train: Training features (numpy array or DataFrame)
    - y_train: Training labels (numpy array or pandas Series)
    - p_train_in_test: Domain classifier probabilities for training samples (numpy array)
    - percentile: Percentile for clipping sample weights (default: 90)
    
    Returns:
    - sample_weights_combined: Combined sample weights (numpy array)
    """
    # Convert to pandas Series if numpy array
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    
    # Compute class weights
    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights_array))
    sample_weights_class = y_train.map(class_weights_dict).values
    
    # Combine class weights with domain classifier probabilities
    sample_weights_combined = sample_weights_class * p_train_in_test
    # Normalize the weights to prevent extreme values
    sample_weights_combined = (sample_weights_combined - np.min(sample_weights_combined)) / (np.max(sample_weights_combined) - np.min(sample_weights_combined) + 1e-10)
    sample_weights_combined = np.clip(sample_weights_combined, 0, np.percentile(sample_weights_combined, percentile))
    
    return sample_weights_combined

# === Step 6: Correct Label Shift ===
def correct_label_shift(y_train, y_test, y_proba, epsilon=1e-2):
    """
    Apply prior probability correction to predicted probabilities to handle label shift.
    
    Parameters:
    - y_train: Training labels (numpy array or pandas Series)
    - y_test: Test labels (numpy array or pandas Series)
    - y_proba: Predicted probabilities (numpy array, shape (n_samples, n_classes))
    - epsilon: Small constant to avoid division by zero (default: 1e-2)
    
    Returns:
    - corrected_proba: Corrected probabilities (numpy array)
    """
    # Convert to pandas Series if numpy array
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    
    num_classes = y_proba.shape[1]
    all_classes = sorted(set(y_train.unique()) | set(y_test.unique()) | set(range(num_classes)))
    
    train_prior = y_train.value_counts(normalize=True).reindex(all_classes, fill_value=epsilon)
    test_prior = y_test.value_counts(normalize=True).reindex(all_classes, fill_value=epsilon)
    
    train_prior = train_prior.reindex(range(num_classes), fill_value=epsilon).values
    test_prior = test_prior.reindex(range(num_classes), fill_value=epsilon).values
    
    correction_factors = test_prior / (train_prior + epsilon)
    # Clip correction factors to avoid extreme values
    correction_factors = np.clip(correction_factors, 0.1, 10.0)
    corrected_proba = y_proba * correction_factors
    corrected_proba /= corrected_proba.sum(axis=1, keepdims=True) + epsilon
    
    return corrected_proba

# === Step 7: Generate Pseudo-Labels ===
def generate_pseudo_labels(X_train_combined, y_train_combined, X_test_unlabeled, confidence_threshold, n_estimators=300, max_depth=12):
    """
    Train a Random Forest model to generate pseudo-labels for unlabeled test data.
    
    Parameters:
    - X_train_combined: Combined training features
    - y_train_combined: Combined training labels
    - X_test_unlabeled: Unlabeled test features
    - confidence_threshold: Confidence threshold for pseudo-labels
    - n_estimators: Number of trees in Random Forest
    - max_depth: Maximum depth of trees
    
    Returns:
    - X_pseudo: Pseudo-labeled features
    - y_pseudo: Pseudo-labels
    - high_conf_count: Number of high-confidence pseudo-labeled samples
    """
    clf_rf_pseudo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf_rf_pseudo.fit(X_train_combined, y_train_combined)
    y_test_prob = clf_rf_pseudo.predict_proba(X_test_unlabeled)
    
    max_probs = y_test_prob.max(axis=1)
    pseudo_labels = y_test_prob.argmax(axis=1)
    high_conf_idx = np.where(max_probs >= confidence_threshold)[0]
    
    X_pseudo = X_test_unlabeled[high_conf_idx]
    y_pseudo = pseudo_labels[high_conf_idx]
    high_conf_count = len(high_conf_idx)
    
    return X_pseudo, y_pseudo, high_conf_count

# === Step 8: Evaluate Model ===
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
    
    # Print total number of samples
    print(f"\nTotal validation samples: {len(y_val)}")

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
    print("Classification Report (All Classes):")
    print(classification_report(y_val, y_pred, labels=np.arange(num_classes), zero_division=0))

    # Plot and save confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    # Simplify the filename
    route_name = name.split()[1].lower()  # e.g., "Route A" -> "route_a"
    save_path = os.path.join(SAVE_PATH, f"confusion_matrix_{route_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Confusion Matrix Saved to {save_path}]")

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "weighted_log_loss": wll
    }

# === Step 9: Route A - Direct Training with Shift Mitigation ===
def route_a(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2):
    """
    Train and evaluate a Random Forest model with covariate and label shift mitigation (Route A).
    
    Parameters:
    - X_train_cleaned: Cleaned training features (numpy array)
    - y_train: Training labels (numpy array)
    - X_test2_labeled_cleaned: Cleaned labeled test features (numpy array)
    - y_test2: Labeled test labels (numpy array)
    """
    print("\n=== Route A: Direct Training with Shift Mitigation ===")
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_cleaned = scaler.fit_transform(X_train_cleaned)
    X_test2_labeled_cleaned = scaler.transform(X_test2_labeled_cleaned)
    
    # Print class distribution for debugging
    print("Training set class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())
    print("Test set class distribution:")
    print(pd.Series(y_test2).value_counts().sort_index())
    
    # Detect domain shift and get probabilities
    auc_mean, auc_std, feature_importances, p_train_in_test = detect_shift_with_domain_classifier(X_train_cleaned, X_test2_labeled_cleaned)
    print(f"[Route A] Domain Classifier AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    
    # Mitigate covariate shift
    sample_weights_combined = mitigate_covariate_shift(X_train_cleaned, y_train, p_train_in_test, percentile=90)
    print("[Route A] Covariate shift mitigated using sample weights")
    
    # Apply SMOTE to handle class imbalance with limited sampling
    class_counts = pd.Series(y_train).value_counts()
    print("Class counts before SMOTE:", class_counts)
    min_class_count = min(class_counts.values)
    k_neighbors = max(1, min(5, min_class_count - 1))
    # Limit the number of samples generated by SMOTE
    sampling_strategy = {label: max(1000, count) for label, count in class_counts.items()}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_cleaned, y_train)
    print(f"[Route A] Applied SMOTE to handle class imbalance. Resampled size: {len(X_train_resampled)}")
    
    # Adjust sample weights for resampled data
    class_counts_resampled = pd.Series(y_train_resampled).value_counts().sort_index()
    class_counts_original = pd.Series(y_train).value_counts().sort_index()
    
    sample_weights_resampled = np.zeros(len(X_train_resampled))
    idx = 0
    for label in class_counts_resampled.index:
        num_resampled = class_counts_resampled[label]
        if label in class_counts_original:
            num_original = class_counts_original[label]
            original_indices = np.where(y_train == label)[0]
            original_weights = sample_weights_combined[original_indices]
            repeat_factor = num_resampled // num_original if num_original > 0 else 1
            remainder = num_resampled % num_original if num_original > 0 else 0
            
            for i in range(len(original_indices)):
                weight = original_weights[i]
                for _ in range(repeat_factor):
                    if idx < len(X_train_resampled):
                        sample_weights_resampled[idx] = weight
                        idx += 1
                if i < remainder and idx < len(X_train_resampled):
                    sample_weights_resampled[idx] = weight
                    idx += 1
        else:
            for _ in range(num_resampled):
                if idx < len(X_train_resampled):
                    sample_weights_resampled[idx] = np.mean(sample_weights_combined)
                    idx += 1
    
    if len(sample_weights_resampled) != len(X_train_resampled):
        raise ValueError(f"sample_weights_resampled length {len(sample_weights_resampled)} does not match X_train_resampled length {len(X_train_resampled)}")
    
    print(f"[Route A] Sample weights adjusted for resampled data. Length: {len(sample_weights_resampled)}")
    
    # Train RandomForest with sample weights
    clf_rf_a = RandomForestClassifier(
        n_estimators=500,  # Increased
        max_depth=16,      # Increased
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf_rf_a.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights_resampled)
    
    # Predict on test set
    y_val_proba = clf_rf_a.predict_proba(X_test2_labeled_cleaned)
    
    # Mitigate label shift
    corrected_proba = correct_label_shift(y_train, y_test2, y_val_proba, epsilon=1e-2)
    y_val_pred = np.argmax(corrected_proba, axis=1)
    print("[Route A] Label shift mitigated using prior probability correction")
    
    # Evaluate
    evaluate_model("Route A", clf_rf_a, None, y_test2, y_val_pred, corrected_proba)

# === Step 10: Route B - Pseudo-Labeling with Default Parameters ===
def route_b(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2, X_test2_unlabeled_cleaned):
    """
    Train and evaluate a Random Forest model with pseudo-labeling (Route B).
    
    Parameters:
    - X_train_cleaned: Cleaned training features
    - y_train: Training labels
    - X_test2_labeled_cleaned: Cleaned labeled test features
    - y_test2: Labeled test labels
    - X_test2_unlabeled_cleaned: Cleaned unlabeled test features
    """
    print("\n=== Route B: Pseudo-Labeling (Default) ===")
    # Standardize the data
    scaler = StandardScaler()
    X_train_cleaned = scaler.fit_transform(X_train_cleaned)
    X_test2_labeled_cleaned = scaler.transform(X_test2_labeled_cleaned)
    X_test2_unlabeled_cleaned = scaler.transform(X_test2_unlabeled_cleaned)
    
    # Combine training and labeled test data
    X_train_combined = np.vstack([X_train_cleaned, X_test2_labeled_cleaned])
    y_train_combined = np.hstack([y_train, y_test2])
    print(f"[Route B] Combined training set size: {X_train_combined.shape[0]} samples")
    
    # Generate pseudo-labels
    X_pseudo, y_pseudo, high_conf_count = generate_pseudo_labels(
        X_train_combined,
        y_train_combined,
        X_test2_unlabeled_cleaned,
        confidence_threshold=0.85,
        n_estimators=500,
        max_depth=16
    )
    print(f"[Route B RF] Selected {high_conf_count} pseudo-labeled samples (confidence >= 0.85)")
    
    # Extend training set
    X_train_full = np.vstack([X_train_combined, X_pseudo])
    y_train_full = np.hstack([y_train_combined, y_pseudo])
    print(f"[Route B] Extended training set size: {X_train_full.shape[0]} samples")
    
    # Split into training and validation (80/20)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    print(f"[Route B] Validation set size: {X_val_split.shape[0]} samples")
    
    # Train final model
    clf_rf_b = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf_rf_b.fit(X_train_split, y_train_split)
    
    # Predict and evaluate
    y_val_prob = clf_rf_b.predict_proba(X_val_split)
    y_val_pred = clf_rf_b.predict(X_val_split)
    evaluate_model("Route B", clf_rf_b, None, y_val_split, y_val_pred, y_val_prob)

# === Step 11: Route B Tweaked - Pseudo-Labeling with Adjusted Parameters ===
def route_b_tweaked(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2, X_test2_unlabeled_cleaned):
    """
    Train and evaluate a Random Forest model with pseudo-labeling and tweaked parameters (Route B Tweaked).
    
    Parameters:
    - X_train_cleaned: Cleaned training features
    - y_train: Training labels
    - X_test2_labeled_cleaned: Cleaned labeled test features
    - y_test2: Labeled test labels
    - X_test2_unlabeled_cleaned: Cleaned unlabeled test features
    """
    print("\n=== Route B Tweaked: Pseudo-Labeling (Adjusted Parameters) ===")
    # Standardize the data
    scaler = StandardScaler()
    X_train_cleaned = scaler.fit_transform(X_train_cleaned)
    X_test2_labeled_cleaned = scaler.transform(X_test2_labeled_cleaned)
    X_test2_unlabeled_cleaned = scaler.transform(X_test2_unlabeled_cleaned)
    
    # Combine training and labeled test data
    X_train_combined = np.vstack([X_train_cleaned, X_test2_labeled_cleaned])
    y_train_combined = np.hstack([y_train, y_test2])
    print(f"[Route B Tweaked] Combined training set size: {X_train_combined.shape[0]} samples")
    
    # Generate pseudo-labels with tweaked parameters
    X_pseudo, y_pseudo, high_conf_count = generate_pseudo_labels(
        X_train_combined,
        y_train_combined,
        X_test2_unlabeled_cleaned,
        confidence_threshold=0.80,
        n_estimators=500,
        max_depth=16
    )
    print(f"[Route B RF - Tweaked] Selected {high_conf_count} pseudo-labeled samples (confidence >= 0.80)")
    
    # Extend training set
    X_train_full = np.vstack([X_train_combined, X_pseudo])
    y_train_full = np.hstack([y_train_combined, y_pseudo])
    print(f"[Route B Tweaked] Extended training set size: {X_train_full.shape[0]} samples")
    
    # Split into training and validation (80/20)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    print(f"[Route B Tweaked] Validation set size: {X_val_split.shape[0]} samples")
    
    # Train final model with tweaked parameters
    clf_rf_b = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf_rf_b.fit(X_train_split, y_train_split)
    
    # Predict and evaluate
    y_val_prob = clf_rf_b.predict_proba(X_val_split)
    y_val_pred = clf_rf_b.predict(X_val_split)
    evaluate_model("Route B Tweaked", clf_rf_b, None, y_val_split, y_val_pred, y_val_prob)

# === Step 12: GridSearchCV - Hyperparameter Tuning ===
def grid_search_route(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2, X_test2_unlabeled_cleaned):
    """
    Perform GridSearchCV to tune Random Forest hyperparameters and evaluate (GridSearch Route).
    
    Parameters:
    - X_train_cleaned: Cleaned training features
    - y_train: Training labels
    - X_test2_labeled_cleaned: Cleaned labeled test features
    - y_test2: Labeled test labels
    - X_test2_unlabeled_cleaned: Cleaned unlabeled test features
    """
    print("\n=== GridSearchCV Route: Hyperparameter Tuning ===")
    # Standardize the data
    scaler = StandardScaler()
    X_train_cleaned = scaler.fit_transform(X_train_cleaned)
    X_test2_labeled_cleaned = scaler.transform(X_test2_labeled_cleaned)
    X_test2_unlabeled_cleaned = scaler.transform(X_test2_unlabeled_cleaned)
    
    # Combine training and labeled test data
    X_train_combined = np.vstack([X_train_cleaned, X_test2_labeled_cleaned])
    y_train_combined = np.hstack([y_train, y_test2])
    print(f"[GridSearch] Combined training set size: {X_train_combined.shape[0]} samples")
    
    # Generate pseudo-labels
    X_pseudo, y_pseudo, high_conf_count = generate_pseudo_labels(
        X_train_combined,
        y_train_combined,
        X_test2_unlabeled_cleaned,
        confidence_threshold=0.80,
        n_estimators=500,
        max_depth=16
    )
    print(f"[GridSearch RF] Selected {high_conf_count} pseudo-labeled samples (confidence >= 0.80)")
    
    # Extend training set
    X_train_full = np.vstack([X_train_combined, X_pseudo])
    y_train_full = np.hstack([y_train_combined, y_pseudo])
    print(f"[GridSearch] Extended training set size: {X_train_full.shape[0]} samples")
    
    # Split into training and validation (80/20)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    print(f"[GridSearch] Validation set size: {X_val_split.shape[0]} samples")
    
    # Perform GridSearchCV
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    f1_macro = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    grid_search = GridSearchCV(rf, param_grid, scoring=f1_macro, cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_split, y_train_split)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best Macro F1 (CV):", grid_search.best_score_)
    
    # Evaluate best model
    best_rf = grid_search.best_estimator_
    y_val_prob = best_rf.predict_proba(X_val_split)
    y_val_pred = best_rf.predict(X_val_split)
    evaluate_model("GridSearch Route", best_rf, None, y_val_split, y_val_pred, y_val_prob)

# === Main Function ===
def main():
    """
    Main function to execute the entire pipeline.
    """
    X_train = pd.read_csv("data/X_train.csv", skiprows=1, header=None).values
    y_train = pd.read_csv("data/y_train.csv", skiprows=1, header=None).values.ravel()
    X_test2 = pd.read_csv("data/X_test_2.csv", skiprows=1, header=None).values
    y_test2 = pd.read_csv("data/y_test_2_reduced.csv", skiprows=1, header=None).values.ravel()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test2 shape: {X_test2.shape}")
    print(f"y_test2 shape: {y_test2.shape}")
    
    # Print test set label distribution
    print("\nTest set label distribution (y_test2):")
    print(pd.Series(y_test2).value_counts().sort_index())
    
    # Step 2: Detect shift with domain classifier
    print("\nDetecting distribution shift with domain classifier...")
    auc_mean, auc_std, feature_importances, p_train_in_test = detect_shift_with_domain_classifier(X_train, X_test2[:202])
    print(f"[RF Domain Classifier] ROC AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    
    # Print top 10 important features
    important_feats_rf = pd.DataFrame({'feature': np.arange(X_train.shape[1]), 'importance': feature_importances})
    important_feats_rf = important_feats_rf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    print("\nTop 10 Important Features (RF Domain Classifier):")
    print(important_feats_rf.head(10))
    
    # Step 3: Detect shift with KS test
    print("\nDetecting covariate shift with KS test...")
    ks_df = detect_shift_with_ks_test(X_train, X_test2[:202])
    print("\nTop 10 Features with Smallest KS p-values:")
    print(ks_df.head(10))
    
    # Step 4: Feature selection
    print("\nSelecting features...")
    X_train_cleaned, X_test2_labeled_cleaned, X_test2_unlabeled_cleaned, final_selected_feats = select_features(
        X_train, X_test2, "output/pairwise_p_value_top_10_features.csv", ks_df, top_k=250
    )
    print(f"X_train_cleaned shape: {X_train_cleaned.shape}")
    print(f"X_test2_labeled_cleaned shape: {X_test2_labeled_cleaned.shape}")
    print(f"X_test2_unlabeled_cleaned shape: {X_test2_unlabeled_cleaned.shape}")
    print(f"Final Selected Features (after removing shift): {len(final_selected_feats)} features")
    
    # Step 5: Route A with shift mitigation
    route_a(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2)
    
    # Step 6: Route B
    route_b(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2, X_test2_unlabeled_cleaned)
    
    # Step 7: Route B Tweaked
    route_b_tweaked(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2, X_test2_unlabeled_cleaned)
    
    # Step 8: GridSearchCV Route
    grid_search_route(X_train_cleaned, y_train, X_test2_labeled_cleaned, y_test2, X_test2_unlabeled_cleaned)

if __name__ == "__main__":
    main()