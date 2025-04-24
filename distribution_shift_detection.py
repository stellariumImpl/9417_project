import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, entropy
import os
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")  # Suppress sklearn warnings
# Global save path for shift detection results
SAVE_PATH = "shift_detection_result"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

from eda import winsorize_iqr  # Import winsorize function from EDA module

def check_covariate_shift_ks_kl(X_train, X_test, alpha=0.05, kl_threshold=0.07, ks_threshold=0.1):
    """
    Detect covariate shift between training and test sets using KS test and KL divergence.

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.
    - alpha (float): Significance level for KS test.
    - kl_threshold (float): Threshold for KL divergence.
    - ks_threshold (float): Threshold for KS statistic.

    Returns:
    - pd.DataFrame: Summary of shift detection per feature.
    """
    ks_results = []

    for col in X_train.columns:
        # KS test
        stat, p_value = ks_2samp(X_train[col], X_test[col])

        # KL divergence (histogram-based)
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

        # Smoothing to avoid division by zero
        train_hist += 1e-8
        test_hist += 1e-8

        kl_div = entropy(train_hist, test_hist)

        # Shift detection flag
        shift_flag = (stat > ks_threshold and p_value < alpha) or (kl_div > kl_threshold)

        ks_results.append({
            "Feature": col,
            "KS_statistic": stat,
            "p_value": p_value,
            "KL_divergence": kl_div,
            "Shift_flag": shift_flag
        })

    return pd.DataFrame(ks_results).sort_values("KS_statistic", ascending=False)

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
    print(f" Domain Classifier AUC: {domain_auc_score:.4f}")

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

    # Optional: clip extreme weights
    sample_weights_combined = np.clip(sample_weights_combined, 0, np.percentile(sample_weights_combined, 95))

    # Train shift-aware model
    model_shift_aware = LGBMClassifier(objective='multiclass', num_class=28, metric='multi_logloss', random_state=42, verbose=-1)
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

    # Combine data
    X_domain = pd.concat([X_train, X_test], axis=0)
    y_domain = np.array([0] * len(X_train) + [1] * len(X_test))

    # Combine weights (train weights + test weights=1)
    domain_weights = np.concatenate([sample_weights_combined, np.ones(len(X_test))])

    # Train domain classifier
    domain_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    domain_clf.fit(X_domain, y_domain, sample_weight=domain_weights)

    # Predict & compute AUC
    y_domain_pred = domain_clf.predict_proba(X_domain)[:, 1]
    domain_auc_post = roc_auc_score(y_domain, y_domain_pred)
    print(f"Domain Classifier AUC after mitigation: {domain_auc_post:.4f}")

    return domain_auc_post

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
    # Ensure Series format
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    labels = sorted(set(y_train.unique()) | set(y_test.unique()))

    # Normalized distributions
    train_dist = y_train.value_counts(normalize=True).reindex(labels, fill_value=0)
    test_dist = y_test.value_counts(normalize=True).reindex(labels, fill_value=0)

    # KL Divergence (Test || Train)
    kl_div = entropy(test_dist + epsilon, train_dist + epsilon)

    # Average absolute difference
    avg_shift = np.abs(train_dist - test_dist).mean()

    # Prepare DataFrame
    shift_df = pd.DataFrame({
        "Train_P(Y)": train_dist,
        "Test_P(Y)": test_dist,
        "Shift_Abs": np.abs(train_dist - test_dist)
    })

    print(shift_df)
    print(f"\nAverage Absolute Shift: {avg_shift:.2%}")
    print(f"KL Divergence (Test || Train): {kl_div:.4f}")

    # Visualization
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

def correct_label_shift_prior(y_train, y_test, y_proba, epsilon=1e-8):
    """
    Correct predicted probabilities using prior correction for label shift.

    Args:
    - y_train (array-like): Training labels.
    - y_test (array-like): Test labels (labeled subset).
    - y_proba (np.ndarray): Model predicted probabilities (n_samples, n_classes).
    - epsilon (float): Smoothing constant.

    Returns:
    - np.ndarray: Corrected probabilities.
    """
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    num_classes = y_proba.shape[1]
    labels = list(range(num_classes))

    # Compute priors
    train_prior = y_train.value_counts(normalize=True).reindex(labels, fill_value=epsilon).values
    test_prior = y_test.value_counts(normalize=True).reindex(labels, fill_value=epsilon).values

    # Apply correction factors
    correction_factors = test_prior / (train_prior + epsilon)
    corrected_proba = y_proba * correction_factors
    corrected_proba /= corrected_proba.sum(axis=1, keepdims=True)

    return corrected_proba



def main():
    try:
        # Load datasets
        X_train = pd.read_csv("data/X_train.csv")
        X_test_2 = pd.read_csv("data/X_test_2.csv")
        y_train = pd.read_csv("data/y_train.csv").squeeze()
        y_test2_reduced = pd.read_csv("data/y_test_2_reduced.csv").squeeze()

        # Winsorization
        X_train = winsorize_iqr(X_train)
        X_test_2 = winsorize_iqr(X_test_2)

        # Step 1: Detect Covariate Shift
        print("\nStep 1: Detect Covariate Shift...")
        result_df_2 = check_covariate_shift_ks_kl(X_train, X_test_2)
        result_df_2.to_csv(os.path.join(SAVE_PATH, "shift_detection_X_test_2.csv"), index=False)
        n_shift_2 = result_df_2['Shift_flag'].sum()
        print(f"X_train vs X_test_2: {n_shift_2}/{len(result_df_2)} features show covariate shift.")

        # Step 2: Mitigate Covariate Shift
        print("\nStep 2: Mitigate Covariate Shift...")
        domain_auc_score, p_train_in_test = detect_domain_shift(X_train, X_test_2)
        domain_output_df = pd.DataFrame({"Sample_index": X_train.index, "Train_in_Test_Prob": p_train_in_test})
        domain_output_df.to_csv(os.path.join(SAVE_PATH, "domain_classifier_output.csv"), index=False)
        model_shift_aware, sample_weights_combined = mitigate_covariate_shift(
            X_train, y_train, p_train_in_test, output_model_path="model_shift_aware.pkl"
        )

        # Step 3: Re-detect Covariate Shift (Validation)
        print("\nStep 3: Validate Covariate Shift Mitigation...")
        validate_domain_shift_mitigation(X_train, X_test_2, sample_weights_combined)

        # Step 4: Detect Label Shift
        print("\nStep 4: Detect Label Shift...")
        label_shift_metrics = detect_label_shift(y_train, y_test2_reduced, save_path=SAVE_PATH)

        # Step 5: Mitigate Label Shift (Prior Correction Example)
        print("\nStep 5: Mitigate Label Shift with Prior Correction...")
        y_proba_test2 = model_shift_aware.predict_proba(X_test_2.iloc[:202])
        corrected_proba = correct_label_shift_prior(y_train, y_test2_reduced, y_proba_test2)

        # Step 6: Re-detect Label Shift (Validation)
        print("\nStep 6: Validate Label Shift Mitigation...")
        corrected_test_prior = corrected_proba.mean(axis=0)
        train_prior = y_train.value_counts(normalize=True).sort_index().values
        test_prior = y_test2_reduced.value_counts(normalize=True).sort_index().reindex(np.arange(corrected_proba.shape[1]), fill_value=0).values

        # Prepare DataFrame for comparison
        validation_df = pd.DataFrame({
            "Train_P(Y)": train_prior,
            "Original_Test_P(Y)": test_prior,
            "Corrected_Test_P(Y)": corrected_test_prior
        })

        print(validation_df)
        print(f"\nAverage Absolute Shift (Corrected vs Train): {np.abs(validation_df['Train_P(Y)'] - validation_df['Corrected_Test_P(Y)']).mean():.2%}")

        # Visualization
        validation_df.plot.bar(figsize=(12, 5))
        plt.title("Label Distribution: Train vs Test (Original & Corrected)")
        plt.ylabel("Proportion")
        plt.xlabel("Label")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, "label_shift_validation.png"))
        plt.close()

        # KL Divergence (Corrected Test || Train)
        kl_div_corrected = entropy(corrected_test_prior + 1e-8, train_prior + 1e-8)
        print(f"KL Divergence (Corrected Test || Train): {kl_div_corrected:.4f}")


    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data files exist in the 'data' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
