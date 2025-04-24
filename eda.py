import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import os

# Global save path for EDA results
SAVE_PATH = "eda_result"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def check_data_quality(X, y):
    """
    Check for missing values and duplicate rows in the feature and target datasets.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series): Target variable.

    Returns:
    - None (prints the results of missing values and duplicates)
    """
    missing_values_X = X.isnull().sum().sum()
    missing_values_y = y.isnull().sum() if isinstance(y, pd.Series) else y.isnull().sum().sum()
    print(f"Number of missing values in feature data: {missing_values_X}")
    print(f"Number of missing values in target data: {missing_values_y}")
    duplicate_rows_X = X.duplicated().sum()
    print(f"Number of duplicate rows in feature data: {duplicate_rows_X}")

def analyze_class_distribution(y, output_filename='class_distribution.png'):
    """
    Analyze and visualize the distribution of classes in the target variable.

    Parameters:
    - y (pd.Series): Target variable containing class labels.
    - output_filename (str): Filename for saving the bar plot (default: 'class_distribution.png').

    Returns:
    - None (prints class counts and saves the bar plot)
    """
    class_counts = y.value_counts()
    print("Class counts:\n", class_counts)
    plt.figure(figsize=(12, 4))
    class_counts.plot(kind='bar')
    plt.title("Number of Samples per Class")
    plt.xlabel("Class Label")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, output_filename)
    plt.savefig(output_path)
    plt.close()

def feature_statistics(X, prefix="feature"):
    """
    Analyze and visualize statistical characteristics of features (mean, std, min, max).

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - prefix (str): Prefix for saved plot filenames (default: 'feature').

    Returns:
    - None (prints feature statistics and saves distribution plots)
    """
    """
    Analyze and visualize statistical characteristics of features (mean, std, min, max).

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - prefix (str): Prefix for saved plot filenames (default: 'feature').

    Returns:
    - None (prints feature statistics and saves distribution plots)
    """
    summary_stats = X.describe().T
    print("Mean, standard deviation, min, and max for each feature:")
    print(summary_stats[['mean', 'std', 'min', 'max']])
    stats_to_plot = [
        ('mean', 'The Mean Distribution of All Features', 'Mean Value'),
        ('std', 'Standard Deviation Distribution of All Features', 'Standard Deviation Value'),
        ('min', 'The Min Distribution of All Features', 'Min Value'),
        ('max', 'The Max Distribution of All Features', 'Max Value')
    ]
    for stat, title, xlabel in stats_to_plot:
        plt.figure(figsize=(8, 4))
        plt.hist(summary_stats[stat], bins=40, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Feature Counts")
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(SAVE_PATH, f"{prefix}_{stat}_distribution.png")
        plt.savefig(output_path)
        plt.close()
    zero_var_cols = summary_stats[summary_stats['std'] == 0].index.tolist()
    low_var_cols = summary_stats[summary_stats['std'] < 0.01].index.tolist()
    print(f"Number of features with zero variance: {len(zero_var_cols)}")
    print(f"Number of features with variance below 0.01: {len(low_var_cols)}")

def plot_high_variance_features(X, num_features=5, prefix="high_variance"):
    """
    Plot the distribution of features with the highest standard deviation.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - num_features (int): Number of top features to plot (default: 5).
    - prefix (str): Prefix for saved plot filenames (default: 'high_variance').

    Returns:
    - None (saves distribution plots in SAVE_PATH)
    """
    sns.set(style="whitegrid")
    top_std_columns = X.std().sort_values(ascending=False).head(num_features).index
    for column in top_std_columns:
        plt.figure(figsize=(20, 6))
        sns.histplot(X[column], bins=30, kde=True, edgecolor='black')
        plt.title(f'Count Distribution of High-Variance Feature: {column}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(SAVE_PATH, f"{prefix}_feature_{column}.png")
        plt.savefig(output_path)
        plt.close()

def analyze_pca_variance(X, prefix="pca"):
    """
    Analyze PCA cumulative explained variance.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - prefix (str): Prefix for saved plot filenames (default: 'pca').

    Returns:
    - None (prints explained variance and saves cumulative variance plot)
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Cumulative explained variance proportion:")
    print(np.cumsum(explained_variance_ratio))

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(explained_variance_ratio), marker='o')
    plt.title("Cumulative Variance Explained (Explained Variance Ratio)")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Proportion")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, f"{prefix}_cumulative_variance.png")
    plt.savefig(output_path)
    plt.close()

def visualize_pca_2d(X, y, prefix="pca_2d"):
    """
    Visualize the 2D projection of the dataset after PCA dimensionality reduction.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series): Target variable.
    - prefix (str): Prefix for saved plot filenames (default: 'pca_2d').

    Returns:
    - None (saves the 2D scatter plot)
    """
    from sklearn.decomposition import PCA

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    X_pca_df = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])
    X_pca_df["label"] = y.values

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("hls", len(X_pca_df["label"].unique()))
    sns.scatterplot(data=X_pca_df, x="PC1", y="PC2", hue="label", palette=palette, legend=True, s=30, alpha=0.6)
    plt.title("Two-Dimensional Projection after PCA (Colored by Category)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, f"{prefix}_scatter.png")
    plt.savefig(output_path)
    plt.close()


def analyze_feature_correlation(X, threshold=0.9, top_n=100, prefix="correlation"):
    """
    Analyze feature-to-feature correlation using Pearson coefficients, visualize heatmap,
    and identify highly correlated feature pairs.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - threshold (float): Correlation threshold for identifying highly correlated pairs (default: 0.9).
    - top_n (int): Number of top features to visualize in heatmap (default: 100).
    - prefix (str): Prefix for saved plot filenames (default: 'correlation').

    Returns:
    - None (prints high correlation pairs and saves heatmap)
    """
    corr_matrix = X.corr()
    subset = X.iloc[:, :top_n]
    corr_subset = subset.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation heatmap of the first {top_n} features (Pearson)")
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, f"{prefix}_heatmap.png")
    plt.savefig(output_path)
    plt.close()

    s_corr_matrix = corr_matrix.abs()
    upper = s_corr_matrix.where(np.triu(np.ones(s_corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, row, s_corr_matrix.loc[row, col])
                       for col in upper.columns
                       for row in upper.index
                       if pd.notnull(upper.loc[row, col]) and upper.loc[row, col] > threshold]

    if high_corr_pairs:
        print(f"Highly correlated feature pairs (correlation > {threshold}):")
        for a, b, c in high_corr_pairs:
            print(f"{a} ↔ {b} → Correlation: {c:.3f}")
    else:
        print(f"No feature pairs found with correlation > {threshold}.")

def detect_outliers_iqr(X, top_n=10, prefix="outlier"):
    """
    Detect and visualize top features with the most outliers using the IQR method.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - top_n (int): Number of top features to visualize (default: 10).
    - prefix (str): Prefix for saved plot filenames (default: 'outlier').

    Returns:
    - None (prints outlier counts and saves boxplots)
    """
    outlier_counts = []
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((X[col] < lower) | (X[col] > upper)).sum()
        outlier_counts.append((col, count))
    outlier_counts_sorted = sorted(outlier_counts, key=lambda x: x[1], reverse=True)
    print("Top features with most outliers:")
    for col, count in outlier_counts_sorted[:top_n]:
        print(f"Feature {col}: Outlier Count = {count}")
    top_outlier_features = [col for col, _ in outlier_counts_sorted[:top_n]]
    plt.figure(figsize=(10, 6))
    X[top_outlier_features].boxplot()
    plt.title("Box plots of top outlier features")
    plt.ylabel("Feature Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, f"{prefix}_boxplot.png")
    plt.savefig(output_path)
    plt.close()

def replace_outliers_iqr(X, threshold=1.5):
    """
    Replace outliers in the dataset using the IQR method and median replacement.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - threshold (float): Multiplier for IQR to define outlier bounds (default: 1.5).

    Returns:
    - pd.DataFrame: Cleaned dataset with outliers replaced by the median.
    """
    X_clean = X.copy()
    for col in X_clean.columns:
        q1 = X_clean[col].quantile(0.25)
        q3 = X_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        median = X_clean[col].median()
        X_clean.loc[(X_clean[col] < lower) | (X_clean[col] > upper), col] = median
    return X_clean

def analyze_skewness_and_normality(X, top_n=10, prefix="normality"):
    """
    Analyze skewness and visualize the top skewed features for normality assessment using KDE and Q-Q plots.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - top_n (int): Number of top skewed features to analyze (default: 10).
    - prefix (str): Prefix for saved plot filenames (default: 'normality').

    Returns:
    - None (prints skewness and saves KDE/Q-Q plots)
    """
    skews = X.skew().abs().sort_values(ascending=False)
    top_skew_features = skews.head(top_n).index.tolist()
    print("Top skewed features:")
    print(skews.head(top_n))

    plt.figure(figsize=(12, 6))
    for col in top_skew_features:
        sns.kdeplot(X[col], label=f"Feature {col}", linewidth=1.5)
    plt.title("KDE of the top skewed features")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(SAVE_PATH, f"{prefix}_kde.png")
    plt.savefig(output_path)
    plt.close()

    plt.figure(figsize=(20, 8))
    for i, col in enumerate(top_skew_features):
        plt.subplot(2, 5, i + 1)
        stats.probplot(X[col], dist="norm", plot=plt)
        plt.title(f"Feature {col}", fontsize=10)
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
    plt.suptitle("Q-Q plot of top skewed features", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = os.path.join(SAVE_PATH, f"{prefix}_qq.png")
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function to execute the full EDA workflow: data quality check, class distribution, feature stats,
    outlier detection and replacement with validation, and normality assessment.

    Steps:
    - Load dataset
    - Check data quality (missing values, duplicates)
    - Analyze class distribution
    - Analyze feature statistics
    - Analyze feature correlation
    - Detect outliers before cleaning
    - Analyze skewness and normality before outlier cleaning
    - Replace outliers using IQR method
    - Validate outliers after cleaning
    - Analyze skewness and normality after outlier cleaning

    - Before and after comparisons are made for outlier detection and normality analysis.

    Returns:
    - None
    """
    try:
        X_train = pd.read_csv("data/X_train.csv")
        y_train = pd.read_csv("data/y_train.csv").squeeze()

        print("Checking data quality...\n")
        check_data_quality(X_train, y_train)

        print("\nAnalyzing class distribution...\n")
        analyze_class_distribution(y_train, 'class_distribution_real.png')

        print("\nAnalyzing feature statistics...\n")
        feature_statistics(X_train, prefix="feature")

        print("\nPlotting high variance feature distributions...\n")
        plot_high_variance_features(X_train, num_features=5, prefix="high_variance")

        print("\nAnalyzing feature correlation...\n")
        analyze_feature_correlation(X_train, threshold=0.9, top_n=100, prefix="correlation")

        print("\nAnalyzing PCA cumulative variance...\n")
        analyze_pca_variance(X_train, prefix="pca")

        print("\nVisualizing PCA 2D projection...\n")
        visualize_pca_2d(X_train, y_train, prefix="pca_2d")

        print("\nDetecting outliers before cleaning...\n")
        detect_outliers_iqr(X_train, top_n=10, prefix="outlier_before")

        print("\nAnalyzing skewness and normality before outlier cleaning...\n")
        analyze_skewness_and_normality(X_train, top_n=10, prefix="normality_before")

        print("\nReplacing outliers with median using IQR method...\n")
        X_clean = replace_outliers_iqr(X_train)

        print("\nRevalidating outliers after cleaning...\n")
        detect_outliers_iqr(X_clean, top_n=10, prefix="outlier_after")

        print("\nAnalyzing skewness and normality after outlier cleaning...\n")
        analyze_skewness_and_normality(X_clean, top_n=10, prefix="normality_after")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data files exist in the 'data' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
