import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tqdm import tqdm
import os

# === Global save path ===
SAVE_PATH = "feature_selection"
os.makedirs(SAVE_PATH, exist_ok=True)

# === Feature Selection Methods ===

def diff(train, label, k):
    train_label = pd.concat([train, label], axis=1)
    per_class_feature_mean = train_label.groupby("label").mean()

    feature_scores = {}
    for feature in per_class_feature_mean.columns:
        class_means = per_class_feature_mean[feature]
        diffs = [abs(class_means[i] - class_means[j]) for i, j in combinations(class_means.index, 2)]
        feature_scores[feature] = np.mean(diffs)

    sorted_features = pd.Series(feature_scores).sort_values(ascending=False)
    return list(map(int, sorted_features.head(k).index.tolist()))

def t_test_diff(train, label, k):
    feature_scores = {}
    class_list = label.unique()
    class_pairs = list(combinations(class_list, 2))

    for col in train.columns:
        p_vals = []
        for c1, c2 in class_pairs:
            group1 = train[label == c1][col]
            group2 = train[label == c2][col]
            _, p_val = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            p_vals.append(p_val)
        feature_scores[col] = np.mean(p_vals)

    sorted_features = pd.Series(feature_scores).sort_values(ascending=True)
    return list(map(int, sorted_features.head(k).index.tolist()))

def select_k_best_mi(train, label, k):
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(train, label)
    scores = selector.scores_
    return np.argsort(scores)[-k:][::-1].tolist()

def fisher_score(train, label, k):
    y = label.values.ravel()
    overall_mean = train.mean(axis=0)
    classes = np.unique(y)

    numerator = np.zeros(train.shape[1])
    denominator = np.zeros(train.shape[1])

    for cls in classes:
        idx = (y == cls)
        n_c = np.sum(idx)
        if n_c == 0:
            continue
        class_mean = train[idx].mean(axis=0)
        class_var = train[idx].var(axis=0) + 1e-8
        numerator += n_c * (class_mean - overall_mean) ** 2
        denominator += n_c * class_var

    fisher_scores = numerator / denominator
    sorted_indices = np.argsort(fisher_scores)[::-1]
    return sorted_indices[:k].tolist()

# === Evaluation Function with Progress Bar ===

def evaluate_k_range(X, y, k_list, feature_selector, name=""):
    results = []
    for k in tqdm(k_list, desc=f"[{name}] Testing different k", position=1, leave=False):
        selected_idx = feature_selector(X, y, k)
        X_selected = X.iloc[:, selected_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            objective='multi:softprob',
            num_class=28,
            eval_metric='mlogloss',
            n_estimators=100,
            random_state=42,
            tree_method='gpu_hist'
        )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        results.append({"k": k, "macro_f1": f1_macro, "weighted_f1": f1_weighted})

    return pd.DataFrame(results)

# === Plotting Functions (no change) ===

def plot_f1_vs_k(all_results):
    plt.figure(figsize=(10, 6))
    for method_name, result_df in all_results.items():
        plt.plot(result_df["k"], result_df["macro_f1"], marker='o', label=f'{method_name} Macro F1')
        plt.plot(result_df["k"], result_df["weighted_f1"], marker='s', label=f'{method_name} Weighted F1')

    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs k (Different Feature Selection Methods)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "f1_vs_k_comparison.png"))

def plot_overlap_matrix(rankings, top_k=100):
    method_names = list(rankings.keys())
    overlap_matrix = np.zeros((len(method_names), len(method_names)), dtype=int)

    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            overlap = len(set(rankings[m1][:top_k]) & set(rankings[m2][:top_k]))
            overlap_matrix[i, j] = overlap

    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, xticklabels=method_names, yticklabels=method_names, cmap="Blues")
    plt.title(f"Feature Overlap Matrix (Top-{top_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"overlap_matrix_top_{top_k}.png"))

def plot_top_features_bar(rankings, method_name, top_n=10):
    top_features = rankings[method_name][:top_n]
    df = pd.DataFrame({"Feature Index": top_features, "Rank": np.arange(1, top_n+1)})
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Rank", y="Feature Index", data=df, palette="viridis")
    plt.title(f"Top-{top_n} Features ({method_name})")
    plt.xlabel("Rank")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"top_{top_n}_features_{method_name.replace(' ', '_').lower()}.png"))

# === Main Execution ===

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv").squeeze()

k_values = [50, 100, 150, 200, 250, 300]
all_results = {}
rankings = {}

methods = {
    "Mutual Info": select_k_best_mi,
    "Mean Diff": diff,
    "T-test": t_test_diff,
    "Fisher Score": fisher_score
}

for method_name, selector in tqdm(methods.items(), desc="Feature Selection Methods", position=0):
    result_df = evaluate_k_range(X, y, k_values, selector, name=method_name)
    all_results[method_name] = result_df
    full_ranking = selector(X, y, k=300)
    rankings[method_name] = full_ranking

plot_f1_vs_k(all_results)
plot_overlap_matrix(rankings, top_k=100)
for method in methods.keys():
    plot_top_features_bar(rankings, method, top_n=10)
