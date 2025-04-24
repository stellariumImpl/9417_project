import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, entropy
import os
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")  # Suppress sklearn warnings

# Global save path for shift detection results
SAVE_PATH = "shift_detection_result"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

from eda import winsorize_iqr  # Import winsorize function from EDA module
from utils import evaluate_model, weighted_log_loss  # Import some useful utils


######################################################
# Covariate Shift Detection and Mitigation Functions #
######################################################

def check_covariate_shift_ks_kl(X_train, X_test, alpha=0.05, kl_threshold=0.07, ks_threshold=0.1):
    """
    Detect covariate shift between training and test sets using KS test and KL divergence.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature naive set.
    - alpha (float): Significance level for KS test.
    - kl_threshold (float): Threshold for KL divergence.
    - ks_threshold (float): Threshold for KS statistic.

    Returns:
    - pd.DataFrame: Summary of shift detection per feature.
    """
    ks_results = []
    for col in X_train.columns:
        if X_train[col].dtype in ['object', 'category']:  # Categorical feature
            from scipy.stats import chi2_contingency
            contingency_table = pd.crosstab(X_train[col], X_test[col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            kl_div = np.nan
            shift_flag = p_value < alpha
        else:  # Numeric feature
            stat, p_value = ks_2samp(X_train[col], X_test[col])
            train_hist, bins = np.histogram(
                X_train[col], 
                bins=50, 
                range=(min(X_train[col].min(), X_test[col].min()), max(X_train[col].max(), X_test[col].max())), 
                density=True
            )
            test_hist, _ = np.histogram(
                X_test[col], 
                bins=bins, 
                density=True
            )
            train_hist += 1e-8
            test_hist += 1e-8
            kl_div = entropy(train_hist, test_hist)
            shift_flag = (stat > ks_threshold and p_value < alpha) or (kl_div > kl_threshold)

        ks_results.append({
            "Feature": col,
            "Statistic": chi2_stat if X_train[col].dtype in ['object', 'category'] else stat,
            "p_value": p_value,
            "KL_divergence": kl_div,
            "Shift_flag": shift_flag
        })

    return pd.DataFrame(ks_results).sort_values("Statistic", ascending=False)

def plot_feature_shift(X_train, X_test, feature_name, prefix="feature_shift"):
    """
    Visualize the distribution shift of a specific feature between training and test sets.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.
    - feature_name (str): Feature name to visualize.
    - prefix (str): Prefix for saved plot filenames.

    Returns:
    - None (saves the KDE plot)
    """
    plt.figure(figsize=(8, 4))
    if X_train[feature_name].dtype in ['object', 'category']:
        sns.countplot(x=feature_name, hue=pd.concat([X_train.assign(Set='Train'), X_test.assign(Set='Test')])['Set'], 
                      data=pd.concat([X_train, X_test]))
    else:
        sns.kdeplot(X_train[feature_name], label="Train", fill=True)
        sns.kdeplot(X_test[feature_name], label="Test", fill=True)
    plt.title(f"Feature Distribution Shift: {feature_name}")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, f"{prefix}_{feature_name}_kde.png")
    plt.savefig(output_path)
    plt.close()

def detect_domain_shift(X_train, X_test):
    """
    Detect covariate shift using a domain classifier (RandomForest) to distinguish train and test samples.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.

    Returns:
    - float: Domain classifier AUC score.
    - np.ndarray: Probability of each training sample being classified as test sample.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X_domain = pd.concat([X_train, X_test], axis=0)
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test))

    domain_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    domain_auc = cross_val_score(domain_clf, X_domain, y_domain, cv=5, scoring='roc_auc')
    domain_auc_score = np.mean(domain_auc)
    print(f"Domain Classifier AUC: {domain_auc_score:.4f}")

    domain_clf.fit(X_domain, y_domain)
    p_train_in_test = domain_clf.predict_proba(X_train)[:, 1]
    return domain_auc_score, p_train_in_test

def mitigate_covariate_shift(X_train, y_train, p_train_in_test, output_model_path="model_shift_aware.pkl"):
    """
    Mitigate covariate shift by combining domain classifier weights with class weights for sample reweighting.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - y_train (pd.Series): Training labels.
    - p_train_in_test (np.ndarray): Domain classifier probabilities for training samples.
    - output_model_path (str): Path to save the shift-aware model.

    Returns:
    - LGBMClassifier: Trained shift-aware model.
    """
    # Compute class weights
    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights_array))
    sample_weights_class = y_train.map(class_weights_dict)

    # Combine class weights with domain classifier probabilities
    sample_weights_combined = sample_weights_class * p_train_in_test
    sample_weights_combined = np.clip(sample_weights_combined, 0, np.percentile(sample_weights_combined, 95))

    # Train shift-aware model
    num_classes = len(np.unique(y_train))
    model_shift_aware = LGBMClassifier(objective='multiclass', num_class=num_classes, metric='multi_logloss', random_state=42, verbose=-1)
    model_shift_aware.fit(X_train, y_train, sample_weight=sample_weights_combined)

    # Save model
    joblib.dump(model_shift_aware, os.path.join(SAVE_PATH, output_model_path))
    print(f"Trained and saved shift-aware model to {output_model_path}")

    return model_shift_aware, sample_weights_combined

def validate_domain_shift_mitigation(X_train, X_test, sample_weights_combined):
    """
    Validate covariate shift mitigation by re-running the domain classifier with sample weights.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.
    - sample_weights_combined (np.ndarray): Combined sample weights (class weight × domain prob).

    Returns:
    - float: Domain classifier AUC after mitigation.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X_domain = pd.concat([X_train, X_test], axis=0)
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test))
    domain_weights = np.concatenate([sample_weights_combined, np.ones(len(X_test))])

    domain_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    domain_clf.fit(X_domain, y_domain, sample_weight=domain_weights)

    y_domain_pred = domain_clf.predict_proba(X_domain)[:, 1]
    domain_auc_post = roc_auc_score(y_domain, y_domain_pred)
    print(f"Domain Classifier AUC after mitigation: {domain_auc_post:.4f}")

    return domain_auc_post

######################################################
#   Label Shift Detection and Mitigation Functions   #
######################################################

def detect_label_shift(y_train, y_test, epsilon=1e-8, visualize=True, save_path="shift_detection_result"):
    """
    Detect label shift between training and test labels using distribution comparison and KL divergence.

    Args:
    - y_train (array-like): Training labels.
    - y_test (array-like): Test labels.
    - epsilon (float): Smoothing constant to avoid division by zero.
    - visualize (bool): Whether to visualize label distributions.
    - save_path (str): Path to save the plots.

    Returns:
    - dict: Contains KL divergence, average shift, and label-wise distribution DataFrame.
    """
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    labels = sorted(set(y_train.unique()) | set(y_test.unique()))

    train_dist = y_train.value_counts(normalize=True).reindex(labels, fill_value=0)
    test_dist = y_test.value_counts(normalize=True).reindex(labels, fill_value=0)

    kl_div = entropy(test_dist + epsilon, train_dist + epsilon)
    avg_shift = np.abs(train_dist - test_dist).mean()

    shift_df = pd.DataFrame({
        "Train_P(Y)": train_dist,
        "Test_P(Y)": test_dist,
        "Shift_Abs": np.abs(train_dist - test_dist)
    })

    print(shift_df)
    print(f"\nAverage Absolute Shift: {avg_shift:.2%}")
    print(f"KL Divergence (Test || Train): {kl_div:.4f}")

    if visualize:
        plt.figure(figsize=(12, 5))
        shift_df[["Train_P(Y)", "Test_P(Y)"]].plot.bar(width=0.8)
        plt.title("Label Distribution: Train vs Test")
        plt.xlabel("Label")
        plt.ylabel("Proportion")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{save_path}/label_shift_distribution.png")
        plt.close()

    return {"kl_divergence": kl_div, "avg_shift": avg_shift, "distribution": shift_df}

def correct_label_shift(y_train, y_test, y_proba, epsilon=1e-6):
    """
    Apply prior probability correction to predicted probabilities to handle label shift.

    Args:
        y_train: array-like, training set labels
        y_test: array-like, test set labels
        y_proba: numpy array, shape (n_samples, n_classes), predicted probabilities
        epsilon: float, small constant to avoid division by zero (default: 1e-6)

    Returns:
        corrected_proba: numpy array, corrected probabilities
    """
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    num_classes = y_proba.shape[1]
    all_classes = sorted(set(y_train.unique()) | set(y_test.unique()) | set(range(num_classes)))
    
    train_prior = y_train.value_counts(normalize=True).reindex(all_classes, fill_value=epsilon)
    test_prior = y_test.value_counts(normalize=True).reindex(all_classes, fill_value=epsilon)
    
    train_prior = train_prior.reindex(range(num_classes), fill_value=epsilon).values
    test_prior = test_prior.reindex(range(num_classes), fill_value=epsilon).values
    
    correction_factors = test_prior / (train_prior + epsilon)
    corrected_proba = y_proba * correction_factors
    corrected_proba /= corrected_proba.sum(axis=1, keepdims=True) + epsilon
    
    return corrected_proba

def validate_label_shift_correction(train_prior, test_prior, corrected_test_prior, save_path=SAVE_PATH):
    """
    Validate and visualize the label shift correction results.
    
    Args:
    - train_prior (array-like): Prior distribution of training labels.
    - test_prior (array-like): Prior distribution of test labels.
    - corrected_test_prior (array-like): Corrected distribution of test labels.
    - save_path (str): Path to save visualization results.
    
    Returns:
    - dict: Contains validation metrics including KL divergence and average shift after correction.
    """
    validation_df = pd.DataFrame({
        "Train_P(Y)": train_prior,
        "Original_Test_P(Y)": test_prior,
        "Corrected_Test_P(Y)": corrected_test_prior
    })

    print(validation_df)
    avg_shift_corrected = np.abs(validation_df['Train_P(Y)'] - validation_df['Corrected_Test_P(Y)']).mean()
    print(f"\nAverage Absolute Shift (Corrected vs Train): {avg_shift_corrected:.2%}")

    validation_df.plot.bar(figsize=(12, 5))
    plt.title("Label Distribution: Train vs Test (Original & Corrected)")
    plt.ylabel("Proportion")
    plt.xlabel("Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "label_shift_validation.png"))
    plt.close()

    kl_div_corrected = entropy(corrected_test_prior + 1e-8, train_prior + 1e-8)
    print(f"KL Divergence (Corrected Test || Train): {kl_div_corrected:.4f}")
    
    return {
        "kl_divergence_corrected": kl_div_corrected,
        "avg_shift_corrected": avg_shift_corrected,
        "validation_df": validation_df
    }



############################
#  Concept Shift Detection #
############################

def analyze_concept_drift(y_true, y_pred):
    """
    Analyze concept drift by identifying the top 10 confused class pairs based on the confusion matrix.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    """
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

def detect_concept_drift(model_initial, X_train, y_train, X_test_2, y_test2_reduced, save_path=SAVE_PATH, performance_threshold=0.01, error_rate_threshold=0.1):
    """
    Detect concept drift by comparing initial and retrained model performance and error patterns.

    Parameters:
    - model_initial: Pretrained model (e.g., shift-aware model).
    - X_train (pd.DataFrame): Training feature set.
    - y_train (pd.Series): Training labels.
    - X_test_2 (pd.DataFrame): Complete test feature set.
    - y_test2_reduced (pd.Series): Labels for part of the test set.
    - save_path (str): Directory to save evaluation results.
    - performance_threshold (float): Minimum F1 Macro improvement to flag concept drift.
    - error_rate_threshold (float): Minimum error rate to consider a class pair significant.

    Returns:
    - dict: Contains performance metrics, error patterns, and concept drift flag.
    """
    # Ensure test set alignment
    X_test_2 = X_test_2.iloc[:len(y_test2_reduced)].reset_index(drop=True)
    y_test2_reduced = y_test2_reduced.reset_index(drop=True)

    # Split test set (50% retrain, 50% evaluate)
    n = len(y_test2_reduced) // 2
    X_retrain, X_eval = X_test_2.iloc[:n], X_test_2.iloc[n:]
    y_retrain, y_eval = y_test2_reduced.iloc[:n], y_test2_reduced.iloc[n:]

    # Validate sample size
    if len(y_eval) < 50:
        print("Warning: Evaluation set too small (<50 samples). Results may be unstable.")
        return {"error": "Insufficient evaluation samples"}

    # Compute label distributions
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    retrain_dist = y_retrain.value_counts(normalize=True).sort_index()
    eval_dist = y_eval.value_counts(normalize=True).sort_index()
    label_dist_df = pd.DataFrame({
        "Train": train_dist,
        "Retrain": retrain_dist,
        "Eval": eval_dist
    }).fillna(0)
    label_dist_df.to_csv(os.path.join(save_path, "concept_drift_label_distributions.csv"))
    print(f"Label distributions saved to {save_path}/concept_drift_label_distributions.csv")

    # Evaluate initial model on evaluation set
    y_pred_initial = model_initial.predict(X_eval)
    y_proba_initial = model_initial.predict_proba(X_eval)

    # Retrain model on train + retrain part
    X_combined = pd.concat([X_train, X_retrain], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_retrain], axis=0).reset_index(drop=True)
    model_retrain = LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y_train)),
        metric='multi_logloss',
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    model_retrain.fit(X_combined, y_combined)

    # Evaluate retrained model
    y_pred_retrain = model_retrain.predict(X_eval)
    y_proba_retrain = model_retrain.predict_proba(X_eval)

    # Compute metrics
    valid_classes = np.unique(y_eval)
    num_classes = max(len(np.unique(y_train)), len(np.unique(y_test2_reduced)))
    y_ohe = label_binarize(y_eval, classes=np.arange(num_classes))
    y_ohe_valid = y_ohe[:, valid_classes]
    y_proba_initial_valid = y_proba_initial[:, valid_classes]
    y_proba_retrain_valid = y_proba_retrain[:, valid_classes]

    metrics = {
        "initial_accuracy": accuracy_score(y_eval, y_pred_initial),
        "initial_f1_macro": f1_score(y_eval, y_pred_initial, average='macro', zero_division=0),
        "initial_weighted_log_loss": weighted_log_loss(y_ohe_valid, y_proba_initial_valid),
        "retrain_accuracy": accuracy_score(y_eval, y_pred_retrain),
        "retrain_f1_macro": f1_score(y_eval, y_pred_retrain, average='macro', zero_division=0),
        "retrain_weighted_log_loss": weighted_log_loss(y_ohe_valid, y_proba_retrain_valid)
    }

    # Compute confusion matrices
    cm_initial = confusion_matrix(y_eval, y_pred_initial, labels=valid_classes)
    cm_retrain = confusion_matrix(y_eval, y_pred_retrain, labels=valid_classes)

    # Extract error pairs
    def get_error_pairs(cm, valid_classes, total_samples):
        pairs = []
        n = cm.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and cm[i][j] > 0:
                    error_rate = cm[i][j] / total_samples
                    pairs.append({
                        "true_class": valid_classes[i],
                        "pred_class": valid_classes[j],
                        "count": cm[i][j],
                        "error_rate": error_rate
                    })
        return sorted(pairs, key=lambda x: x["count"], reverse=True)[:10]

    total_samples = len(y_eval)
    initial_error_pairs = get_error_pairs(cm_initial, valid_classes, total_samples)
    retrain_error_pairs = get_error_pairs(cm_retrain, valid_classes, total_samples)

    # Detect significant error pattern changes
    error_changes = []
    for pair_initial in initial_error_pairs:
        for pair_retrain in retrain_error_pairs:
            if (pair_initial["true_class"] == pair_retrain["true_class"] and
                pair_initial["pred_class"] == pair_retrain["pred_class"]):
                error_rate_change = pair_initial["error_rate"] - pair_retrain["error_rate"]
                if abs(error_rate_change) > error_rate_threshold:
                    error_changes.append({
                        "true_class": pair_initial["true_class"],
                        "pred_class": pair_initial["pred_class"],
                        "initial_error_rate": pair_initial["error_rate"],
                        "retrain_error_rate": pair_retrain["error_rate"],
                        "change": error_rate_change
                    })

    # Flag concept drift
    f1_improvement = metrics["retrain_f1_macro"] - metrics["initial_f1_macro"]
    concept_drift_flag = f1_improvement > performance_threshold or len(error_changes) > 0

    # Visualize confusion matrices
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_initial, annot=True, fmt='d', cmap='Blues', xticklabels=valid_classes, yticklabels=valid_classes)
    plt.title("Initial Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_retrain, annot=True, fmt='d', cmap='Blues', xticklabels=valid_classes, yticklabels=valid_classes)
    plt.title("Retrained Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "concept_drift_confusion_matrices.png"))
    plt.close()

    # Visualize error rate changes
    if error_changes:
        error_change_df = pd.DataFrame(error_changes)
        error_change_df["pair"] = error_change_df.apply(lambda x: f"{x['true_class']}→{x['pred_class']}", axis=1)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=error_change_df, x="change", y="pair", palette="coolwarm")
        plt.title("Significant Error Rate Changes (Initial - Retrain)")
        plt.xlabel("Error Rate Change")
        plt.ylabel("Class Pair (True→Pred)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "concept_drift_error_changes.png"))
        plt.close()

    # Print results
    print("\nConcept Drift Detection Results:")
    print(f"Initial Model - Accuracy: {metrics['initial_accuracy']:.4f}, F1 Macro: {metrics['initial_f1_macro']:.4f}, Weighted Log Loss: {metrics['initial_weighted_log_loss']:.4f}")
    print(f"Retrained Model - Accuracy: {metrics['retrain_accuracy']:.4f}, F1 Macro: {metrics['retrain_f1_macro']:.4f}, Weighted Log Loss: {metrics['retrain_weighted_log_loss']:.4f}")
    print(f"F1 Macro Improvement: {f1_improvement:.4f}")
    print(f"Concept Drift Detected: {concept_drift_flag}")
    if error_changes:
        print("\nSignificant Error Pattern Changes:")
        for change in error_changes:
            print(f"Class {change['true_class']}→{change['pred_class']}: Initial Error Rate={change['initial_error_rate']:.4f}, Retrain Error Rate={change['retrain_error_rate']:.4f}, Change={change['change']:.4f}")

    return {
        "metrics": metrics,
        "initial_error_pairs": initial_error_pairs,
        "retrain_error_pairs": retrain_error_pairs,
        "error_changes": error_changes,
        "concept_drift_flag": concept_drift_flag,
        "label_dist_df": label_dist_df
    }

def main():
    try:
        # Load datasets
        X_train = pd.read_csv("data/X_train.csv")
        X_test_2 = pd.read_csv("data/X_test_2.csv")
        y_train = pd.read_csv("data/y_train.csv").squeeze()
        y_test2_reduced = pd.read_csv("data/y_test_2_reduced.csv").squeeze()

        # Validate data
        if X_train.shape[1] != X_test_2.shape[1]:
            raise ValueError("X_train and X_test_2 have different number of features.")
        if len(y_test2_reduced) > len(X_test_2):
            raise ValueError("y_test2_reduced has more samples than X_test_2.")

        # Winsorization
        X_train = winsorize_iqr(X_train)
        X_test_2 = winsorize_iqr(X_test_2)

        # Step 1: Detect Covariate Shift
        print("\nStep 1: Detect Covariate Shift...\n")
        result_df_2 = check_covariate_shift_ks_kl(X_train, X_test_2)
        result_df_2.to_csv(os.path.join(SAVE_PATH, "shift_detection_X_test_2.csv"), index=False)
        n_shift_2 = result_df_2['Shift_flag'].sum()
        print(f"X_train vs X_test_2: {n_shift_2}/{len(result_df_2)} features show covariate shift.")

        # Step 2: Mitigate Covariate Shift
        print("\nStep 2: Mitigate Covariate Shift...\n")
        domain_auc_score, p_train_in_test = detect_domain_shift(X_train, X_test_2)
        domain_output_df = pd.DataFrame({"Sample_index": X_train.index, "Train_in_Test_Prob": p_train_in_test})
        domain_output_df.to_csv(os.path.join(SAVE_PATH, "domain_classifier_output.csv"), index=False)
        model_shift_aware, sample_weights_combined = mitigate_covariate_shift(
            X_train, y_train, p_train_in_test, output_model_path="model_shift_aware.pkl"
        )

        # Step 3: Validate Covariate Shift Mitigation
        print("\nStep 3: Validate Covariate Shift Mitigation...\n")
        validate_domain_shift_mitigation(X_train, X_test_2, sample_weights_combined)

        # Step 4: Detect Label Shift
        print("\nStep 4: Detect Label Shift...\n")
        label_shift_metrics = detect_label_shift(y_train, y_test2_reduced, save_path=SAVE_PATH)

        # Step 5: Mitigate Label Shift with Prior Correction
        print("\nStep 5: Mitigate Label Shift with Prior Correction...\n")
        X_test2_labeled_sel = X_test_2.iloc[:len(y_test2_reduced)]  # Align with y_test2_reduced
        y_proba_shift = model_shift_aware.predict_proba(X_test2_labeled_sel)
        y_pred_shift = model_shift_aware.predict(X_test2_labeled_sel)

        # Evaluate before correction
        evaluate_model("Shift-Aware Model (Before Correction)", model_shift_aware, X_test2_labeled_sel, y_test2_reduced, y_pred_shift, y_proba_shift)

        # Apply label shift correction
        corrected_proba = correct_label_shift(y_train, y_test2_reduced, y_proba_shift)
        corrected_pred = np.argmax(corrected_proba, axis=1)  # Update predictions based on corrected probabilities

        # Evaluate after correction
        evaluate_model("Shift-Aware Model (Label-Shift Corrected)", model_shift_aware, X_test2_labeled_sel, y_test2_reduced, corrected_pred, corrected_proba)

        # Step 6: Validate Label Shift Mitigation
        print("\nStep 6: Validate Label Shift Mitigation...\n")
        corrected_test_prior = corrected_proba.mean(axis=0)
        train_prior = y_train.value_counts(normalize=True).sort_index().reindex(range(corrected_proba.shape[1]), fill_value=0).values
        test_prior = y_test2_reduced.value_counts(normalize=True).sort_index().reindex(range(corrected_proba.shape[1]), fill_value=0).values
        validate_label_shift_correction(train_prior, test_prior, corrected_test_prior)

        # Step 7: Analyze Concept Drift
        print("\nStep 7: Analyze Concept Drift...\n")
        analyze_concept_drift(y_test2_reduced, corrected_pred)

        # Step 8: Detect Concept Drift
        print("\nStep 8: Detect Concept Drift...\n")
        detect_concept_drift(
            model_shift_aware, X_train, y_train, X_test_2, y_test2_reduced
        )


    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data files exist in the 'data' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()