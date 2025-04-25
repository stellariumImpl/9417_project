import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# methods of feature selection

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

# evaluate_k_range function

def evaluate_k_range(X, y, k_list, feature_selector, name=""):
    results = []

    for k in k_list:
        print(f"\n>> Testing k = {k} using {name}")
        selected_idx = feature_selector(X, y, k)
        X_selected = X.iloc[:, selected_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        results.append({
            "k": k,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted
        })

    result_df = pd.DataFrame(results)
    return result_df

# Visualization function

def plot_f1_vs_k(result_df, method_name):
    plt.figure(figsize=(10, 5))
    plt.plot(result_df["k"], result_df["macro_f1"], marker='o', label='Macro F1')
    plt.plot(result_df["k"], result_df["weighted_f1"], marker='s', label='Weighted F1')
    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score vs k ({method_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").squeeze()

# Top k needs to be selected
k_values = [50, 100, 150, 200, 250, 300]

# Test SelectKBest + MI
mi_result = evaluate_k_range(X, y, k_values, select_k_best_mi, name="SelectKBest + MI")
print("\n=== Results: SelectKBest (Mutual Information) ===")
print(mi_result)
plot_f1_vs_k(mi_result, "SelectKBest + Mutual Information")

