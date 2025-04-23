# =============================
#  Imports
# =============================
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import json

matplotlib.use("TkAgg")  # Set backend to ensure plots are displayed correctly in some environments

# =============================
#  Data Loading & Preprocessing
# =============================
# Load training data
x_train = pd.read_csv('datasets/X_train.csv')
y_train = pd.read_csv('datasets/y_train.csv').squeeze()

# Describe value ranges for EDA
x_train_range = x_train.describe()
y_train_range = y_train.describe()

# Identify the top 3 features with the highest standard deviation
top_std_columns = x_train.std().sort_values(ascending=False).head(3).index

# Replace outliers using IQR method
def replace_outliers_iqr(df, threshold=1.5):
    df_clean = df.copy()
    for col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        median = df_clean[col].median()

        df_clean.loc[(df_clean[col] < lower) | (df_clean[col] > upper), col] = median

    return df_clean

# Apply outlier removal to the training data
x_train_cleaned = replace_outliers_iqr(x_train)

# =============================
#  Feature Selection
# =============================
"""
Feature Selection Process (Three-Way Agreement Method)

To enhance model performance and reduce dimensionality, we implemented a multi-criteria feature
selection strategy combining three distinct techniques:

1. Mutual Information (MI): Quantifies the dependency between individual features and the target
   class. We retained features with an MI score ≥ 0.0034.

2. Random Forest Importance (RF): Based on feature importance derived from a balanced RandomForest
   model. Features with an importance score ≥ 0.0024 were selected.

3. Inter-Class Mean Difference (MeanDiff): Measures how different feature means vary across classes. 
Features with a class-wise standard deviation ≥ 0.34 were considered relevant.

To ensure stability and generalizability, only features that satisfied all three criteria were selected 
(i.e., set intersection of the above methods).

Additionally, to gain insight into the cumulative contribution of features ranked by 
Random Forest importance, we calculated the number of top-ranked features required to account 
for 90% of the total importance. This threshold helps evaluate model sparsity and justifies the 
selected feature count. This analysis revealed that only the top *k* features (denoted `k_90`) 
are responsible for the vast majority of the predictive power.

The final feature set was then used consistently across model training and evaluation to 
ensure alignment with the feature space learned during training.
"""

# Method 1: Mutual Information (MI)
mi_scores = mutual_info_classif(x_train, y_train.squeeze())
mi_series = pd.Series(mi_scores, index=x_train.columns).sort_values(ascending=False)
top_mi_features = mi_series.head(20)
top_mi_features.plot(kind='bar', title='Top 20 Features by Mutual Information')

# Method 2: Random Forest Feature Importance (RF)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(x_train, y_train.values.ravel())
rf_importance = pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
rf_importance.head(20).plot(kind='bar', title='Top 20 Features by Random Forest Importance')

# Method 3: Inter-Class Mean Difference (MeanDiff)
class_means = x_train.groupby(y_train).mean()
mean_diff = class_means.std(axis=0)
mean_diff_sorted = mean_diff.sort_values(ascending=False)
mean_diff_sorted.head(20).plot(kind='bar', color='steelblue')

# Determine how many features account for 90% of RF importance
rf_sorted = rf_importance.sort_values(ascending=False)
cumulative_importance = np.cumsum(rf_sorted)
k_90 = np.argmax(cumulative_importance >= 0.9) + 1

# #Plot Cumulative Feature Importance Curve (Random Forest)
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o', color='green')
# plt.axhline(y=0.9, color='red', linestyle='--', label='90% Total Importance')
# plt.axvline(x=k_90, color='blue', linestyle='--', label=f'Top {k_90} Features')
# plt.title("Cumulative Feature Importance (Random Forest)")
# plt.xlabel("Number of Top Features")
# plt.ylabel("Cumulative Importance")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("cumulative_rf_importance_curve.png", dpi=300)
# plt.close()

# # =============================
# # Visualize Distributions of Feature Selection Scores
# # =============================
# # --- 1. Mutual Information Score Distribution ---
# # Shows the distribution of MI scores across all features
# # A red vertical line indicates the threshold used for selection (e.g., 0.04)
mi_series = mi_series.sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# sns.histplot(mi_series, bins=30, kde=True, color='skyblue')
# plt.axvline(x=0.03, color='red', linestyle='--', label='Threshold: 0.04')
# plt.title("Distribution of Mutual Information Scores")
# plt.xlabel("Mutual Information Score")
# plt.ylabel("Feature Count")
# plt.legend()
# plt.tight_layout()
# plt.savefig("mutual_info_distribution_threshold.png", dpi=300)
# plt.show()
#
# # --- 2. Cross-Class Mean Difference Distribution ---
# # Plots the standard deviation of feature means across classes
# # Features with high variance are considered more discriminative
# plt.figure(figsize=(10, 6))
# sns.histplot(mean_diff_sorted, bins=30, kde=True, color='skyblue')
# plt.axvline(x=0.4, color='red', linestyle='--', label='Threshold: 0.4')
# plt.title("Distribution of Cross-Class Mean Differences")
# plt.xlabel("Std of Class Means (Feature Importance)")
# plt.ylabel("Feature Count")
# plt.tight_layout()
# plt.show()
#
#
# # --- 3. Random Forest Feature Importance Distribution ---
# # Displays how RF distributed importance across features
# # Red line marks the selection threshold
rf_series = rf_importance.sort_values(ascending=False)
# plt.figure(figsize=(10, 6))
# sns.histplot(rf_series, bins=30, kde=True, color='green')
# plt.axvline(x=0.0023, color='red', linestyle='--', label='Threshold: 0.0025')
# plt.title("Distribution of Random Forest Feature Importances")
# plt.xlabel("Random Forest Importance Score")
# plt.ylabel("Feature Count")
# plt.legend()
# plt.tight_layout()
# plt.savefig("random_forest_importance_distribution.png", dpi=300)
# plt.show()


# Select features that are above thresholds in all three methods
selected_mi_features = mi_series[mi_series >= 0.0034].index
selected_rf_features = rf_series[rf_series >= 0.0024].index
selected_md_features = mean_diff_sorted[mean_diff_sorted >= 0.34].index

# Intersection of selected features across all methods
selected_features = list(set(selected_mi_features) & set(selected_rf_features) & set(selected_md_features))

# Final feature subset
x_selected = x_train[selected_features]
print(f"selected_features: {selected_features}")
print(f"The numebr of selected_features: {len(selected_features)}")

# Save selected features
with open("selected_features.json", "w") as f:
    json.dump(selected_features, f)

# Visualize Feature Selection Overlap using Venn Diagram
venn3(
    (set(selected_mi_features), set(selected_rf_features), set(selected_md_features)),
    set_labels=('Mutual Info', 'Random Forest', 'Mean Diff')
)
plt.title("Feature Selection Method Overlap")
plt.show()

# =============================
#     Train-Test Split
# =============================

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_selected, y_train, test_size=0.3, stratify=y_train, random_state=42)

# =============================
#  Evaluation Function
# =============================
def evaluate_model(name, model, X_val, y_val, y_pred, y_proba):
    classes = np.unique(y_val)
    y_ohe = label_binarize(y_val, classes=classes)

    def weighted_log_loss(y_true, y_pred):
        class_counts = np.sum(y_true, axis=0)
        class_weights = 1.0 / class_counts
        class_weights /= np.sum(class_weights)
        sample_weights = np.sum(y_true * class_weights, axis=1)
        loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        return loss

    acc_val = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    log_loss_val = log_loss(y_ohe, y_proba)
    weighted_loss = weighted_log_loss(y_ohe, y_proba)

    print(f"\n Classification Report - {name}")
    print(classification_report(y_val, y_pred, zero_division=0))

    print(f"\n Evaluation Metrics - {name}")
    print(f"{'Accuracy:':<25}{acc_val:.4f}")
    print(f"{'F1 Macro:':<25}{f1_macro:.4f}")
    print(f"{'F1 Weighted:':<25}{f1_weighted:.4f}")
    print(f"{'Log Loss:':<25}{log_loss_val:.4f}")
    print(f"{'Weighted Log Loss:':<25}{weighted_loss:.4f}")

    # Save confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fig_path = f"conf_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fig_path)
    print(f" Confusion matrix saved to {fig_path}")
    plt.close()

# =============================
#  Model 1: Plain XGBoost
# =============================
"""
Model 1: Plain XGBoost (Baseline Model)

This model serves as the baseline for our multi-class classification task. It uses a standard
XGBoost classifier without any techniques to handle class imbalance. All training samples are
treated equally, and no class or sample weights are applied during training.

As a baseline model, this setup allows us to benchmark the performance of more advanced
techniques, such as cost-sensitive learning. By comparing its results against other models,
we can clearly quantify the improvements achieved through imbalance-aware strategies.

Performance is evaluated using standard metrics (Accuracy, F1-score, Log Loss), as well as a
custom weighted log loss that penalizes misclassification of rare classes more heavily.
"""


# Initialize a basic XGBoost classifier for multi-class classification
# - 'multi:softprob' outputs class probabilities
# - num_class = 28 (number of classes in our target)
# - eval_metric = 'mlogloss' is suitable for probabilistic multi-class output
clf_xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=28,
    eval_metric='mlogloss',
    random_state=42
)

# Fit the model on the training set without any class balancing (no sample_weight passed)
clf_xgb.fit(x_train_split, y_train_split)

# Generate predictions (discrete class labels) on the validation set
y_pred_plain = clf_xgb.predict(x_val_split)

# Generate predicted probabilities (needed for log loss calculation)
y_proba_plain = clf_xgb.predict_proba(x_val_split)

# Redefine the custom weighted log loss function (also used in evaluation)
# This penalizes mistakes on minority classes more heavily
def weighted_log_loss(y_true, y_pred):
    class_counts = np.sum(y_true, axis=0)          # Get number of samples per class
    class_weights = 1.0 / class_counts             # Inverse frequency weighting
    class_weights /= np.sum(class_weights)         # Normalize weights to sum to 1
    sample_weights = np.sum(y_true * class_weights, axis=1)  # Per-sample weight
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
    return loss

# Evaluate model performance on validation set using predefined metrics:
# accuracy, F1 scores, log loss, weighted log loss, and confusion matrix
evaluate_model("XGBoost (No Balancing)", clf_xgb, x_val_split, y_val_split, y_pred_plain, y_proba_plain)


# =============================
#  Model 2: Cost-Sensitive XGBoost
# =============================
"""
Model 2: Cost-Sensitive XGBoost

In this model, we address the issue of class imbalance by applying cost-sensitive learning.
Instead of treating all training samples equally, we assign higher weights to minority class
samples and lower weights to majority class samples. These weights are calculated based on
the inverse frequency of each class in the training set.

By passing these weights to the XGBoost model via the `sample_weight` parameter during training,
the model becomes more sensitive to under-represented categories, improving recall and F1-score
for minority classes without sacrificing performance on majority ones.
"""

#  Compute class weights based on class frequencies
# Rare classes receive higher weights to combat class imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_split),
    y=y_train_split
)
class_weights_dict = dict(zip(np.unique(y_train_split), class_weights_array))

# Map class weights to individual training samples
sample_weights = y_train_split.map(class_weights_dict)

# Initialize XGBoost classifier for multi-class classification
clf_xgb_cost = XGBClassifier(
    objective='multi:softprob',   # Predict probabilities for each class
    num_class=28,                 # Total number of target classes
    eval_metric='mlogloss',      # Use multi-class log loss for evaluation
    random_state=42
)

# Fit the model with cost-sensitive sample weights
clf_xgb_cost.fit(x_train_split, y_train_split, sample_weight=sample_weights)

# Predict labels on the validation set
y_pred_cost = clf_xgb_cost.predict(x_val_split)
y_proba_cost = clf_xgb_cost.predict_proba(x_val_split)

# Evaluate model performance using validation set
evaluate_model("XGBoost (Cost-Sensitive)", clf_xgb_cost, x_val_split, y_val_split, y_pred_cost, y_proba_cost)


# =============================
#  Model 3: XGBoost with Hyperparameter Tuning (Cost-sensitive + Optimized)
# =============================
"""
Model 3: XGBoost with Hyperparameter Tuning (Cost-sensitive + Optimized)

This model extends the cost-sensitive XGBoost by introducing randomized hyperparameter search
to further improve performance. It aims to outperform both the baseline and cost-sensitive
versions by finding the best combination of hyperparameters through a more extensive
exploration of the model space.

Key enhancements include:
- Cost-sensitive learning via per-sample weights derived from inverse class frequency
- Hyperparameter tuning using RandomizedSearchCV with cross-validation
- Optimization objective: weighted F1-score, which better captures performance in imbalanced settings

The search is conducted over a diverse set of parameters, including number of estimators,
tree depth, learning rate, subsampling rate, and regularization terms (L1 and L2).

This model combines class imbalance handling with model complexity control, and serves as a
strong candidate for final deployment if generalization performance is validated on unseen data.
"""

# Compute class weights to handle class imbalance
# - Less frequent classes are assigned higher weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_split),
    y=y_train_split
)

# Map weights to class labels
class_weights_dict = dict(zip(np.unique(y_train_split), class_weights_array))

# Create sample-wise weights for training
sample_weights = y_train_split.map(class_weights_dict)

# Define the hyperparameter search space for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],         # Number of boosting rounds
    'max_depth': [3, 5, 7, 9],                         # Tree depth (model complexity)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],           # Step size shrinkage
    'subsample': [0.7, 0.8, 0.9, 1.0],                 # Fraction of samples used per tree
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],          # Fraction of features used per tree
    'reg_alpha': [0, 0.1, 1, 5],                       # L1 regularization term
    'reg_lambda': [1, 5, 10]                           # L2 regularization term
}

# Initialize base XGBoost model
clf = XGBClassifier(
    objective='multi:softprob',     # Output class probabilities
    num_class=28,                   # Total number of classes
    eval_metric='mlogloss',        # Use multi-class log loss as the evaluation metric
    random_state=42
)

# Set up randomized search for hyperparameter tuning
rs = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_dist,    # Search space
    scoring='f1_weighted',             # Optimize for weighted F1-score
    cv=3,                              # 3-fold cross-validation
    n_iter=35,                         # Number of parameter settings sampled
    verbose=2,                         # Verbose output for tracking progress
    n_jobs=-1,                         # Use all available cores
    random_state=42
)

# Fit the model using weighted training samples
rs.fit(x_train_split, y_train_split, sample_weight=sample_weights)

# Retrieve the best model from the search
best_model = rs.best_estimator_

# Predict class labels and probabilities on the validation set
y_pred_best = best_model.predict(x_val_split)
y_proba_best = best_model.predict_proba(x_val_split)

# Evaluate performance using accuracy, F1 scores, log loss, and confusion matrix
evaluate_model("XGBoost (Tuned + Weights)", best_model, x_val_split, y_val_split, y_pred_best, y_proba_best)


# =============================
#  Save the Best Model
# =============================
import joblib

# Save the best XGBoost model obtained from hyperparameter tuning
# The model includes optimized parameters and learned weights
joblib.dump(best_model, "best_xgboost_model.pkl")
print("Best XGBoost model saved to best_xgboost_model.pkl")

# =============================
#  Save the Best Model Predictions for Test Set 1
# =============================

# Load Test Set 1 (unlabeled, shape should be [1000, 300])
# This test set contains customer feedback features only.
X_test_1 = pd.read_csv("datasets/X_test_1.csv")
X_test_1_selected = X_test_1[selected_features]

# Load the previously saved best XGBoost model
# This model was trained using cost-sensitive learning and hyperparameter tuning
best_model = joblib.load("best_xgboost_model.pkl")


# Use the model to predict class probabilities on the test set
# The output shape should be [1000, 28] as required by the submission format
preds_1 = best_model.predict_proba(X_test_1_selected)

# Save the predicted probabilities to a NumPy file (.npy)
np.save("preds_1.npy", preds_1)
print(f"preds_1.npy saved. Shape: {preds_1.shape}")

# =============================
#  Validation Data Format
# =============================
print(np.load("preds_1.npy").shape)  # should be (1000, 28)

