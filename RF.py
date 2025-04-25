import warnings
warnings.filterwarnings("ignore")  # Suppress sklearn warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import os

# Global save path
SAVE_PATH = "rf_before_shift"
os.makedirs(SAVE_PATH, exist_ok=True)

from utils import weighted_log_loss  # Import some useful utils

def select_features_by_pairwise_mean_diff(X, y, top_k=10, save_path=SAVE_PATH):
    """
    Select features based on pairwise mean differences between classes.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target labels.
    - top_k (int): Number of top features to select for each class pair (default: 10).
    - save_path (str): Directory to save output CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing feature indices and their frequency across class pairs.
    """
    # Get unique classes and generate all class pairs
    class_list = np.unique(y)
    class_pairs = list(combinations(class_list, 2))

    # Dictionary to store top-k features for each class pair
    pairwise_top_k_mean_diff = {}

    # Calculate mean difference for each class pair
    for cls0, cls1 in class_pairs:
        idx0 = (y == cls0)
        idx1 = (y == cls1)

        # Compute mean of features for each class
        mean0 = X[idx0].mean(axis=0)
        mean1 = X[idx1].mean(axis=0)
        diff = np.abs(mean0 - mean1)

        # Select top-k features based on mean difference
        top_k_idx_mean_diff = np.argsort(diff)[::-1][:top_k]

        # Store feature information
        feature_info_mean_diff = []
        for rank, feat_idx in enumerate(top_k_idx_mean_diff):
            feature_info_mean_diff.append({
                "feature_idx": int(feat_idx),
                "mean_diff": float(diff[feat_idx]),
                "rank": rank + 1
            })

        pairwise_top_k_mean_diff[(int(cls0), int(cls1))] = feature_info_mean_diff

    # Collect all selected feature indices
    all_top_k_features_mean_diff = []
    for pair in pairwise_top_k_mean_diff.values():
        all_top_k_features_mean_diff.extend([f['feature_idx'] for f in pair])

    # Count frequency of each feature
    feature_counter_mean_diff = Counter(all_top_k_features_mean_diff)

    # Convert to DataFrame
    pairwise_mean_diff_top_k_features_df = pd.DataFrame(
        feature_counter_mean_diff.items(),
        columns=["feature_idx", "count"]
    )
    pairwise_mean_diff_top_k_features_df.sort_values(by="count", ascending=False, inplace=True)

    # Save DataFrame to CSV
    output_path = os.path.join(save_path, "pairwise_mean_diff_top_10_features.csv")
    pairwise_mean_diff_top_k_features_df.to_csv(output_path, index=False)
    print(f"Saved pairwise mean diff feature frequency DataFrame to {output_path}")

    return pairwise_mean_diff_top_k_features_df

def select_features_by_max_class_mean_diff(X, y, save_path=SAVE_PATH):
    """
    Select features based on maximum mean difference across all classes.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target labels.
    - save_path (str): Directory to save output CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing feature indices and their maximum mean differences.
    """
    # Get unique classes
    class_list = np.unique(y)

    # Store list of max class mean differences
    mean_diff_list = []

    # Traverse all features
    for feature in range(X.shape[1]):
        # Compute mean values of the feature for each class
        means = [X[y == label].iloc[:, feature].mean() for label in class_list]

        # Calculate maximum mean difference for the current feature
        max_diff = np.max(means) - np.min(means)

        mean_diff_list.append((feature, max_diff))

    # Convert to DataFrame
    max_class_mean_diff_df = pd.DataFrame(mean_diff_list, columns=["feature_idx", "mean_diff"])
    max_class_mean_diff_df.sort_values(by="mean_diff", ascending=False, inplace=True)

    # Save DataFrame to CSV
    output_path = os.path.join(save_path, "max_class_mean_diff.csv")
    max_class_mean_diff_df.to_csv(output_path, index=False)
    print(f"Saved max class mean diff DataFrame to {output_path}")

    return max_class_mean_diff_df

def select_features_by_pairwise_p_value(X, y, top_k=10, save_path=SAVE_PATH):
    """
    Select features based on pairwise t-test p-values between classes.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target labels.
    - top_k (int): Number of top features to select for each class pair (default: 10).
    - save_path (str): Directory to save output CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing feature indices and their frequency across class pairs.
    """
    # Get unique classes and generate all class pairs
    class_list = np.unique(y)
    class_pairs = list(combinations(class_list, 2))

    # Dictionary to store top-k features for each class pair
    pairwise_top_k_p_value = {}

    # Perform t-test for each class pair
    for cls0, cls1 in class_pairs:
        idx0 = (y == cls0)
        idx1 = (y == cls1)

        # Compute p-values for each feature
        p_values = []
        for feature in range(X.shape[1]):
            stat, p = ttest_ind(
                X[idx0].iloc[:, feature],
                X[idx1].iloc[:, feature],
                equal_var=False,
                nan_policy='omit'
            )
            p_values.append(p)

        # Convert to numpy array
        p_values = np.array(p_values)

        # Select top-k features with smallest p-values
        top_k_idx_p_value = np.argsort(p_values)[:top_k]

        # Store feature information
        feature_info_p_value = []
        for rank, feat_idx in enumerate(top_k_idx_p_value):
            feature_info_p_value.append({
                "feature_idx": int(feat_idx),
                "p_value": float(p_values[feat_idx]),
                "rank": rank + 1
            })

        pairwise_top_k_p_value[(int(cls0), int(cls1))] = feature_info_p_value

    # Collect all selected feature indices
    all_top_k_features_p_value = []
    for pair in pairwise_top_k_p_value.values():
        all_top_k_features_p_value.extend([f['feature_idx'] for f in pair])

    # Count frequency of each feature
    feature_counter_p_value = Counter(all_top_k_features_p_value)

    # Convert to DataFrame
    pairwise_p_value_top_k_features_df = pd.DataFrame(
        feature_counter_p_value.items(),
        columns=["feature_idx", "count"]
    )
    pairwise_p_value_top_k_features_df.sort_values(by="count", ascending=False, inplace=True)

    # Save DataFrame to CSV
    output_path = os.path.join(save_path, "pairwise_p_value_top_10_features.csv")
    pairwise_p_value_top_k_features_df.to_csv(output_path, index=False)
    print(f"Saved pairwise p-value feature frequency DataFrame to {output_path}")

    return pairwise_p_value_top_k_features_df

def select_features_by_mutual_info(X, y, discrete_features=False, random_state=42, save_path=SAVE_PATH):
    """
    Select features based on mutual information scores between features and labels.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target labels.
    - discrete_features (bool): Whether features are discrete (default: False).
    - random_state (int): Random seed for reproducibility (default: 42).
    - save_path (str): Directory to save output CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing feature indices and their mutual information scores.
    """
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)

    # Construct DataFrame of mutual information scores
    mutual_info_df = pd.DataFrame({
        "feature_idx": np.arange(len(mi_scores)),
        "mi_global_score": mi_scores
    })
    mutual_info_df.sort_values(by="mi_global_score", ascending=False, inplace=True)

    # Save DataFrame to CSV
    output_path = os.path.join(save_path, "mutual_info_global.csv")
    mutual_info_df.to_csv(output_path, index=False)
    print(f"Saved mutual information DataFrame to {output_path}")

    return mutual_info_df

def select_features_by_anova_f_test(X, y, top_k=10, save_path=SAVE_PATH):
    """
    Select features based on ANOVA F-test scores and p-values.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target labels.
    - top_k (int): Number of top features to display (default: 10).
    - save_path (str): Directory to save output CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing feature indices, F-scores, and p-values.
    """
    # Perform ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=X.shape[1])
    selector.fit(X, y)

    # Extract F-scores, p-values, and feature indices
    f_scores = selector.scores_
    p_values = selector.pvalues_
    feature_idx = np.arange(X.shape[1])

    # Construct DataFrame
    f_score_with_p_value_df = pd.DataFrame({
        'feature_idx': feature_idx,
        'f_score': f_scores,
        'p_value': p_values
    })
    f_score_with_p_value_df.sort_values(by='f_score', ascending=False, inplace=True)

    # Save DataFrame to CSV
    output_path = os.path.join(save_path, "f_score_with_p_value.csv")
    f_score_with_p_value_df.to_csv(output_path, index=False)
    print(f"Saved ANOVA F-score and p-value DataFrame to {output_path}")

    # Display top-k features
    print(f"\nTop {top_k} selected features by ANOVA F-score and p-value:")
    print(f_score_with_p_value_df.head(top_k))

    return f_score_with_p_value_df

def plot_anova_f_scores(f_score_df, top_k=10, save_path=SAVE_PATH):
    """
    Plot a horizontal bar chart of the top-k ANOVA F-scores.

    Parameters:
    - f_score_df (pd.DataFrame): DataFrame containing feature_idx and f_score columns.
    - top_k (int): Number of top features to plot (default: 10).
    - save_path (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 5))
    top_k_df = f_score_df.head(top_k)
    plt.barh(
        [f"F{int(i)}" for i in top_k_df["feature_idx"]][::-1],  # Reverse for top-down order
        top_k_df["f_score"][::-1],
        color='skyblue'
    )
    plt.xlabel("ANOVA F-score")
    plt.ylabel("Feature Index")
    plt.title(f"Top {top_k} Features by ANOVA F-score")
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(save_path, "anova_f_scores.png")
    plt.savefig(output_path)
    print(f"Saved ANOVA F-score plot to {output_path}")
    plt.show()

def load_feature_indices(filepath, column="feature_idx"):
    """
    Load feature indices from a CSV file.

    Parameters:
    - filepath (str): Path to the CSV file.
    - column (str): Column name containing feature indices (default: "feature_idx").

    Returns:
    - list: List of feature indices.
    """
    try:
        df = pd.read_csv(filepath)
        if column in df.columns:
            return df[column].astype(int).tolist()
        else:
            return df.iloc[:, 0].astype(int).tolist()
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Skipping this ranking.")
        return []

def evaluate_topk_f1_rf_manualcv(X, y, feature_rankings, k_list=None, cv=5, save_path=SAVE_PATH):
    """
    Evaluate RandomForest performance on top-k features using stratified cross-validation and SMOTE.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target labels.
    - feature_rankings (dict): Dictionary of feature ranking lists, keyed by method name.
    - k_list (list): List of top-k feature counts to evaluate (default: [50, 100, 150, 200, 250]).
    - cv (int): Number of cross-validation folds (default: 5).
    - save_path (str): Directory to save output CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing F1 scores for each method and k value.
    """
    if k_list is None:
        k_list = [50, 100, 150, 200, 250]

    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, ranking in tqdm(feature_rankings.items(), desc="Evaluating Sequences"):
        if not ranking:  # Skip empty rankings
            continue
        for k in tqdm(k_list, desc=f"{name} Top-K", leave=False):
            # Select top-k features
            selected_features = ranking[:min(k, len(ranking))]
            X_k = X[:, selected_features]
            f1_scores = []

            # Perform cross-validation
            for train_idx, val_idx in skf.split(X_k, y):
                X_train_cv, X_val_cv = X_k[train_idx], X_k[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                # Ensure SMOTE is safe
                class_counts = Counter(y_train_cv)
                min_class_count = min(class_counts.values())
                k_safe = max(1, min(5, min_class_count - 1))

                # Filter classes with enough samples
                valid_classes = [cls for cls, cnt in class_counts.items() if cnt >= k_safe + 1]
                mask = np.isin(y_train_cv, valid_classes)
                X_train_safe, y_train_safe = X_train_cv[mask], y_train_cv[mask]

                # Apply SMOTE
                try:
                    sampler = SMOTE(random_state=42, k_neighbors=k_safe)
                    X_resampled, y_resampled = sampler.fit_resample(X_train_safe, y_train_safe)
                except ValueError:
                    X_resampled, y_resampled = X_train_safe, y_train_safe

                # Train RandomForest
                model = RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_resampled, y_resampled)

                # Evaluate
                y_pred = model.predict(X_val_cv)
                f1 = f1_score(y_val_cv, y_pred, average="weighted", zero_division=0)
                f1_scores.append(f1)

            # Store results
            results.append({
                "method": name,
                "k": k,
                "f1_score": np.mean(f1_scores)
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)


    return results_df

def report_best_topk(results_df):
    """
    Report the best top-k and corresponding F1 score for each feature selection method.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing method, k, and f1_score columns.

    Result:
    Best Top-K per feature sequence:
        f_score              -> Top-150, Weighted F1 = 0.7283
        mutual_info          -> Top-250, Weighted F1 = 0.7304
        max_mean_diff        -> Top-250, Weighted F1 = 0.7278
        pairwise_mean        -> Top-200, Weighted F1 = 0.7301
        pairwise_pval        -> Top-250, Weighted F1 = 0.7323
    """
    print("\nBest Top-K per feature sequence:")
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        best_row = method_data.loc[method_data['f1_score'].idxmax()]
        print(f"{method:20s} -> Top-{int(best_row['k']):3d}, Weighted F1 = {best_row['f1_score']:.4f}")


def plot_topk_f1_curves(results_df, save_path=SAVE_PATH):
    """
    Plot weighted F1-score curves for different feature selection methods.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing method, k, and f1_score columns.
    - save_path (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        plt.plot(method_data['k'], method_data['f1_score'], marker='o', label=method)

    plt.xlabel("Top-K Features")
    plt.ylabel("Weighted F1-score")
    plt.title("Top-K Feature Subset Evaluation (Random Forest + SMOTE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(save_path, "f1_curves.png")
    plt.savefig(output_path)
    print(f"Saved F1 score curves to {output_path}")

def plot_feature_overlap_matrix(feature_rankings, top_k=100, save_path=SAVE_PATH):
    """
    Plot a heatmap of the overlap matrix among top-k features for different methods.

    Parameters:
    - feature_rankings (dict): Dictionary of feature ranking lists, keyed by method name.
    - top_k (int): Number of top features to consider for overlap (default: 100).
    - save_path (str): Directory to save the plot.
    """
    print(f"\n[INFO] Overlap Matrix Among Top-{top_k} Features")
    rank_names = list(feature_rankings.keys())
    overlap_matrix = np.zeros((len(rank_names), len(rank_names)), dtype=int)

    # Calculate overlap between each pair of feature rankings
    for i, a in enumerate(rank_names):
        for j, b in enumerate(rank_names):
            overlap_matrix[i, j] = len(set(feature_rankings[a][:top_k]) & set(feature_rankings[b][:top_k]))

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, xticklabels=rank_names, yticklabels=rank_names, cmap="Blues")
    plt.title(f"Top-{top_k} Feature Overlap Matrix")
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(save_path, "feature_overlap_matrix.png")
    plt.savefig(output_path)
    print(f"Saved feature overlap matrix to {output_path}")

def train_final_model(X, y, feature_rankings, best_method, best_k, validation_size=2000, save_path=SAVE_PATH):
    """
    Train a final RandomForest model using the best feature selection method and top-k features,
    evaluate on a separate validation set of specified size, and report metrics for classes 0-27.

    Results:
    Training final model using pairwise_pval with top-250 features...
    Training set size: 8000 samples
    Validation set size: 2000 samples

    Overall Classification Report (Classes 0-27):
                precision    recall  f1-score   support

            0     0.0000    0.0000    0.0000         4
            1     0.0000    0.0000    0.0000         1
            2     0.0000    0.0000    0.0000         1
            3     0.5714    0.3077    0.4000        13
            4     0.5208    0.5208    0.5208        48
            5     0.8742    0.9542    0.9125       896
            6     0.8443    0.9279    0.8841       111
            7     0.7273    0.3810    0.5000        21
            8     0.6250    0.8252    0.7113       103
            9     0.0000    0.0000    0.0000         5
            10     0.6908    0.8380    0.7573       216
            11     0.4211    0.5000    0.4571        16
            12     0.5051    0.5495    0.5263        91
            13     0.0000    0.0000    0.0000        12
            14     0.1364    0.0566    0.0800        53
            15     0.8333    1.0000    0.9091         5
            16     0.0000    0.0000    0.0000         1
            17     0.8780    0.5070    0.6429        71
            18     0.7500    0.2500    0.3750        12
            19     0.8571    0.3429    0.4898        35
            20     0.2667    0.1290    0.1739        31
            21     0.7115    0.6852    0.6981        54
            22     0.0000    0.0000    0.0000         1
            23     1.0000    0.3750    0.5455         8
            24     0.4054    0.3896    0.3974        77
            25     0.4400    0.2973    0.3548        37
            26     0.6279    0.4821    0.5455        56
            27     0.8947    0.8095    0.8500        21

        accuracy                         0.7535      2000
    macro avg     0.4850    0.3974    0.4190      2000
    weighted avg     0.7299    0.7535    0.7321      2000


    [SUMMARY] Average Metrics
    Macro F1-score:    0.4190
    Weighted F1-score: 0.7321
    Weighted Log Loss: 0.0062
    """
    print(f"\nTraining final model using {best_method} with top-{best_k} features...")
    
    # Split data into training and validation sets
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X, y,
        test_size=validation_size,
        stratify=y,
        random_state=42
    )
    print(f"Training set size: {X_train_sub.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    
    # Select top-k features
    top_features = feature_rankings[best_method][:best_k]
    X_train_sub_top = X_train_sub[:, top_features]
    X_val_top = X_val[:, top_features]
    
    # Apply SMOTE to handle class imbalance
    class_counts = Counter(y_train_sub)
    min_class_count = min(class_counts.values())
    k_neighbors = max(1, min(5, min_class_count - 1))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train_sub_top, y_train_sub)
    
    # Train RandomForest model
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_resampled, y_resampled)
    
    # Predict on validation set
    y_pred = model.predict(X_val_top)
    y_prob = model.predict_proba(X_val_top)
    
    # Calculate metrics
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    
    # Convert y_val to one-hot encoding for weighted log loss
    y_true_onehot = np.eye(28)[y_val]
    
    # Compute weighted log loss using the custom function
    logloss_weighted = weighted_log_loss(y_true_onehot, y_prob)
    
    # Generate classification report for classes 0-27
    print("\nOverall Classification Report (Classes 0-27):")
    print(classification_report(
        y_val, y_pred,
        labels=np.arange(0, 28),
        digits=4,
        zero_division=0
    ))
    
    # Summary metrics
    print("\n[SUMMARY] Average Metrics")
    print(f"Macro F1-score:    {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print(f"Weighted Log Loss: {logloss_weighted:.4f}")


def main():
    """
    Main function to perform feature selection, evaluate RandomForest performance, and plot results.
    """
    try:
        # Load datasets
        print("\nLoading data...")
        X_train = pd.read_csv("data/X_train.csv")
        y_train = pd.read_csv("data/y_train.csv").squeeze()

        # Validate data
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train have different number of samples.")

        # Perform feature selection using pairwise mean differences
        print("\nPerforming feature selection based on pairwise mean differences...")
        pairwise_mean_diff_df = select_features_by_pairwise_mean_diff(X_train, y_train, top_k=10, save_path=SAVE_PATH)

        # Perform feature selection using max class mean differences
        print("\nPerforming feature selection based on max class mean differences...")
        max_class_mean_diff_df = select_features_by_max_class_mean_diff(X_train, y_train, save_path=SAVE_PATH)

        # Perform feature selection using pairwise t-test p-values
        print("\nPerforming feature selection based on pairwise t-test p-values...")
        pairwise_p_value_df = select_features_by_pairwise_p_value(X_train, y_train, top_k=10, save_path=SAVE_PATH)

        # Perform feature selection using mutual information
        print("\nPerforming feature selection based on mutual information...")
        mutual_info_df = select_features_by_mutual_info(X_train, y_train, discrete_features=False, random_state=42, save_path=SAVE_PATH)

        # Perform feature selection using ANOVA F-test
        print("\nPerforming feature selection based on ANOVA F-test...")
        f_score_with_p_value_df = select_features_by_anova_f_test(X_train, y_train, top_k=10, save_path=SAVE_PATH)

        # Load feature rankings
        print("\nLoading feature rankings...")
        feature_rankings = {
            "pairwise_mean": load_feature_indices(os.path.join(SAVE_PATH, "pairwise_mean_diff_top_10_features.csv")),
            "max_mean_diff": load_feature_indices(os.path.join(SAVE_PATH, "max_class_mean_diff.csv")),
            "pairwise_pval": load_feature_indices(os.path.join(SAVE_PATH, "pairwise_p_value_top_10_features.csv")),
            "mutual_info": load_feature_indices(os.path.join(SAVE_PATH, "mutual_info_global.csv")),
            "f_score": load_feature_indices(os.path.join(SAVE_PATH, "f_score_with_p_value.csv"))
        }

        # # Preprocess data for RandomForest
        print("\nPreprocessing data for RandomForest evaluation...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train_array = y_train.values

        # Evaluate RandomForest with cross-validation
        print("\nEvaluating RandomForest performance...")
        rf_results_df = evaluate_topk_f1_rf_manualcv(
            X_train_scaled,
            y_train_array,
            feature_rankings,
            k_list=[50, 100, 150, 200, 250],
            cv=5,
            save_path=SAVE_PATH
        )

        # Display RandomForest results
        print("\nRandomForest F1 scores:")
        print(rf_results_df)

        # Plot F1 curves
        print("\nPlotting F1 score curves...")
        plot_topk_f1_curves(rf_results_df, save_path=SAVE_PATH)

        # Train final model using the best method and top-k on a validation set of 2000 samples
        best_method = "pairwise_pval"  # Based on previous best F1 score
        best_k = 250  # Based on previous best top-k
        print("\nTraining final model...")
        train_final_model(
            X_train_scaled,
            y_train_array,
            feature_rankings,
            best_method,
            best_k,
            validation_size=2000,
            save_path=SAVE_PATH
        )

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data files exist in the 'data' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()