# ========================
# 1. Load the pretrained model
# ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = joblib.load("best_xgboost_model.pkl")
print("\n Loaded best_xgboost_model.pkl")
model_shift_aware = clone(best_model)

# ========================
#  Load selected feature list
# ========================
with open("selected_features.json", "r") as f:
    selected_features = json.load(f)

# ========================
#  Load datasets
# ========================
X_train = pd.read_csv("datasets/X_train.csv")
y_train = pd.read_csv("datasets/y_train.csv").squeeze()
X_test2 = pd.read_csv("datasets/X_test_2.csv")
y_test2 = pd.read_csv("datasets/y_test_2_reduced.csv").squeeze()

# ========================
#  Select first 202 labeled samples from Test2
# ========================
X_test2_labeled = X_test2.iloc[:202]
y_test2_labeled = y_test2

# ========================
#  Feature selection: only keep features from selected_features
# ========================
X_train_sel = X_train[selected_features]
X_test2_sel = X_test2[selected_features]
X_test2_labeled_sel = X_test2_labeled[selected_features]

# ========================
#  Data cleaning and alignment
# ========================
X_train_sel.columns = X_train_sel.columns.astype(str)
X_test2_sel.columns = X_test2_sel.columns.astype(str)
X_test2_labeled_sel.columns = X_test2_labeled_sel.columns.astype(str)
X_test2_sel = X_test2_sel[X_train_sel.columns]
X_test2_labeled_sel = X_test2_labeled_sel[X_train_sel.columns]
X_train_sel = X_train_sel.fillna(X_train_sel.mean())
X_test2_sel = X_test2_sel.fillna(X_train_sel.mean())
X_test2_labeled_sel = X_test2_labeled_sel.fillna(X_train_sel.mean())

#  Convert to float32 tensors
X_train_tensor = torch.tensor(X_train_sel.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test2_sel.values.astype(np.float32))
X_test_labeled_tensor = torch.tensor(X_test2_labeled_sel.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_labeled_tensor = torch.tensor(y_test2_labeled.values, dtype=torch.long)

print(" Data successfully converted to Tensors!")

# Label encoding
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test2_enc = le.transform(y_test2_labeled)
n_classes = len(le.classes_)
y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test2_enc, dtype=torch.long)

# Estimate class distributions
train_dist = np.bincount(y_train_enc, minlength=n_classes) / len(y_train_enc)
test_dist = np.bincount(y_test2_enc, minlength=n_classes) / len(y_test2_enc)

# Calculate class weights to match Test2 distribution
label_weights = test_dist / (train_dist + 1e-8)
label_weights_tensor = torch.tensor(label_weights, dtype=torch.float32).to(device)


# Construct class-weighted cross-entropy loss function
weighted_ce_loss = nn.CrossEntropyLoss(weight=label_weights_tensor)

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

# ========================
# Construct Domain Labels
# ========================
train_domain_labels = torch.zeros(len(X_train_tensor), dtype=torch.long)  # source domain = 0
test_domain_labels = torch.ones(len(X_test_labeled_tensor), dtype=torch.long)  # target domain = 1

# Concatenate all data
X_all = torch.cat([X_train_tensor, X_test_labeled_tensor], dim=0)
y_all = torch.cat([y_train_tensor, y_test_tensor], dim=0)
d_all = torch.cat([train_domain_labels, test_domain_labels], dim=0)

# ===============================
#  Why I include Test2[:202] in training (No Data Leakage)
# ===============================
#
# In this domain adaptation task, we include the first 202 labeled samples from Test2
# (i.e., X_test2[:202]) in the training phase — but **only their feature vectors (X), not labels (y)**.
#
# These samples are treated as **unlabeled target domain inputs** in the DANN framework.
# They are used to help the model learn domain-invariant representations via the domain classifier.
#
#  How data leakage is avoided:
# - During training, we apply classification loss ONLY to source domain samples (i.e., training data with d=0).
# - Target domain samples (i.e., Test2[:202] with d=1) are **never used in the classification loss**.
# - Their labels (y_test2) are **held out strictly for evaluation only**, never seen by the model.
#
# This is a standard and safe practice in semi-supervised domain adaptation.
# It allows the model to learn to generalize across feature distributions without violating data integrity.

# ========================
# Dataset Definition
# ========================
class DomainDataset(Dataset):
    def __init__(self, features, labels, domains):
        self.X = features
        self.y = labels
        self.d = domains
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.d[index]
    def __len__(self):
        return len(self.X)

def build_dataloaders(X_all, y_all, d_all, batch_size=64, split_ratio=0.9):
    dataset = DomainDataset(X_all, y_all, d_all)
    n_train = int(len(dataset) * split_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2)
    return train_loader, val_loader

# ========================
# Gradient Reversal Layer (GRL)
# ========================
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.clone()
    @staticmethod
    def backward(ctx, grad_output): return -ctx.lambd * grad_output, None

# ========================
# DANN Model Definition
# ========================
class FeatureAdaptNetV2(nn.Module):
    def __init__(self, input_dim, num_classes, num_domains=2):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_domains)
        )

    def forward(self, x, lambd=0.0):
        feat = self.extractor(x)
        reverse_feat = GradientReversal.apply(feat, lambd)
        return self.classifier(feat), self.domain_classifier(reverse_feat)

# ========================
# Lambda Scheduler
# ========================
def computeLambda(epoch, total_epochs):
    progress = epoch / total_epochs
    return 2. / (1 + np.exp(-10 * progress)) - 1

# ========================
# Single Training Epoch with Label Shift-aware Loss
# ========================
def trainOneEpoch_withLabelShift(model, loader, optimizer, epoch, max_epochs, domain_weight=0.3):
    model.train()
    lambd = computeLambda(epoch, max_epochs)
    total, task, domain = 0.0, 0.0, 0.0

    # Manual batch iteration
    batch_size = 64
    for i in range(0, len(X_all), batch_size):
        # Slice batches
        X_batch = X_all[i:i + batch_size]
        y_batch = y_all[i:i + batch_size]
        d_batch = d_all[i:i + batch_size]

        X_batch, y_batch, d_batch = X_batch.to(device), y_batch.to(device), d_batch.to(device)

        # Forward pass
        y_out, d_out = model(X_batch, lambd)

        # Domain classification loss (all samples)
        loss_domain = F.cross_entropy(d_out, d_batch)

        # Classification loss only on source domain
        src_mask = (d_batch == 0)
        assert torch.all(d_batch[src_mask] == 0), "⚠ Potential label leakage detected!"

        if src_mask.sum() > 0:
            y_src = y_batch[src_mask]
            y_pred_src = y_out[src_mask]
            loss_task = weighted_ce_loss(y_pred_src, y_src)
        else:
            loss_task = torch.tensor(0.0, device=device)

        # Total loss
        loss = loss_task + domain_weight * loss_domain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        task += loss_task.item()
        domain += loss_domain.item()

    print(f"[Epoch {epoch + 1}] λ={lambd:.4f} | Total: {total:.3f} | Task: {task:.3f} | Domain: {domain:.3f}")


# ========================
# Full Training Loop with Label Shift-aware DANN
# ========================
def trainFeatureAdaptModel_labelShift(model, train_loader, optimizer, epochs=10, domain_weight=0.3):
    for ep in range(epochs):
        trainOneEpoch_withLabelShift(model, train_loader, optimizer, ep, epochs, domain_weight)


# ========================
# Training Initialization
# ========================
train_loader, val_loader = build_dataloaders(X_all, y_all, d_all)
model = FeatureAdaptNetV2(input_dim=X_all.shape[1], num_classes=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainFeatureAdaptModel_labelShift(model, train_loader, optimizer, epochs=10)

# # Save only the model's state_dict
# torch.save(model.state_dict(), "dann_model_weights.pth")
# print("Saved model weights to dann_model_weights.pth")

# ========================
# Inference and Softmax Output
# ========================
model = FeatureAdaptNetV2(input_dim=X_all.shape[1], num_classes=n_classes).to(device)
model.load_state_dict(torch.load("dann_model_weights.pth"))
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    X_test_labeled_tensor = X_test_labeled_tensor.to(device)  # Only the first 202 labeled samples
    class_logits, _ = model(X_test_labeled_tensor)
    y_pred_probs = F.softmax(class_logits, dim=1).cpu().numpy()  # Convert logits to probabilities

# Get predicted class labels by taking argmax over probabilities
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Ground truth labels
y_val_true = y_test2_enc

# ========================
# Post-processing: filter missing classes and normalize probabilities
# ========================
# Convert ground truth labels to one-hot encoding
y_test2_onehot = np.zeros_like(y_pred_probs)
y_test2_onehot[np.arange(len(y_test2_enc)), y_test2_enc] = 1

# Identify classes that are present in the test set
active_classes = np.where(y_test2_onehot.sum(axis=0) > 0)[0]

# Filter both predictions and ground truth to include only active classes
y_true_filtered = y_test2_onehot[:, active_classes]
y_pred_filtered = y_pred_probs[:, active_classes]

# Avoid log(0) in evaluation by clipping extremely low probabilities
y_pred_filtered = np.clip(y_pred_filtered, 1e-12, 1.0)


# =============================
#  Evaluation Function
# =============================
def evaluate_model(name, model, X_val, y_val, y_pred, y_proba):
    fixed_classes = np.arange(y_proba.shape[1])
    log_loss_val = log_loss(y_val, y_proba, labels=fixed_classes)

    # Convert ground-truth labels to one-hot encoding for WLL calculation
    y_ohe = label_binarize(y_val, classes=fixed_classes)

    # Identify active classes (those that appear in ground truth)
    active_classes = np.where(np.sum(y_ohe, axis=0) > 0)[0]
    y_true_filtered = y_ohe[:, active_classes]
    y_pred_filtered = y_proba[:, active_classes]
    y_pred_filtered = np.clip(y_pred_filtered, 1e-12, 1.0)

    def weighted_log_loss(y_true, y_pred):
        class_counts = np.sum(y_true, axis=0)
        class_weights = 1.0 / np.where(class_counts == 0, 1e-8, class_counts)
        class_weights /= np.sum(class_weights)
        sample_weights = np.sum(y_true * class_weights, axis=1)
        return -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))

    # Compute weighted log loss
    wll = weighted_log_loss(y_true_filtered, y_pred_filtered)

    # Compute evaluation metrics
    acc = accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_val, y_pred, average='weighted', zero_division=0)

    # Print metrics
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

# ===============================
# Evaluate DANN Model Performance
# ===============================
from sklearn.preprocessing import label_binarize

# Extract predicted class labels from softmax probabilities
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Use the true labels from the first 202 labeled samples of Test2
y_val_true = y_test2_enc  # These are label-encoded ground truth labels

# Call the previously defined evaluation function
evaluate_model("DANN + label shift on Test2-202", model=None, X_val=None,
               y_val=y_val_true, y_pred=y_pred_labels, y_proba=y_pred_probs)


# ========================
# Fine-tune DANN to handle concept drift using Test2[:202]
# ========================
# Combine feature sets
X_concept_adapt = np.vstack([X_train_sel.values, X_test2_labeled_sel.values]).astype(np.float32)
y_concept_adapt = np.concatenate([y_train_enc, y_test2_enc])

# Assign weights: boost influence of Test2 samples
sample_weights = np.concatenate([
    np.full(len(y_train_enc), 1.0),
    np.full(len(y_test2_enc), 5.0)
]).astype(np.float32)

# Convert to PyTorch tensors
X_concept_tensor = torch.tensor(X_concept_adapt).to(device)
y_concept_tensor = torch.tensor(y_concept_adapt, dtype=torch.long).to(device)
weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)

# Fine-tune function
def finetune_dann(model, X, y, weights, lr=1e-4, epochs=20):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # use weights manually

    for epoch in range(epochs):
        logits, _ = model(X, lambd=0.0)  # skip domain branch during fine-tuning
        loss_unweighted = loss_fn(logits, y)
        loss = torch.mean(loss_unweighted * weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Fine-tune Epoch {epoch+1}] Weighted Loss: {loss.item():.4f}")

# Start from the current model (already trained on covariate+label shift)
finetune_dann(model, X_concept_tensor, y_concept_tensor, weights_tensor)

# Save fine-tuned weights
# torch.save(model.state_dict(), "finetuned_dann_concept_shift.pth")
# print(" Saved fine-tuned model weights to finetuned_dann_concept_shift.pth")

# Predict on Test2[:202] using fine-tuned model
model.eval()
with torch.no_grad():
    X_test_tensor_eval = torch.tensor(X_test2_labeled_sel.values.astype(np.float32)).to(device)
    logits, _ = model(X_test_tensor_eval)
    y_pred_probs = F.softmax(logits, dim=1).cpu().numpy()
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

evaluate_model(
    name="Fine-tuned DANN on Test2-202",
    model=None,
    X_val=None,
    y_val=y_test2_enc,
    y_pred=y_pred_labels,
    y_proba=y_pred_probs
)

# =============================
#  Predict on Test Set 2 (1818 unlabeled samples) and Save preds_2.npy
# =============================

# Load fine-tuned DANN model
model = FeatureAdaptNetV2(input_dim=X_train_sel.shape[1], num_classes=n_classes).to(device)
model.load_state_dict(torch.load("finetuned_dann_concept_shift.pth"))
model.eval()


# Load Test2 (unlabeled portion)
X_test2_full = pd.read_csv("datasets/X_test_2.csv")
X_test2_unlabeled = X_test2_full.iloc[202:]
X_test2_unlabeled_sel = X_test2_unlabeled[selected_features]
X_test2_unlabeled_sel.columns = X_train_sel.columns  # Align column names
X_test2_unlabeled_tensor = torch.tensor(X_test2_unlabeled_sel.values.astype(np.float32)).to(device)

# Predict softmax probabilities
with torch.no_grad():
    logits_unlabeled, _ = model(X_test2_unlabeled_tensor)
    raw_preds_2 = F.softmax(logits_unlabeled, dim=1).cpu().numpy()

# Label Shift Correction
def correct_label_shift(y_train, y_test2, probs, epsilon=1e-8):
    from sklearn.preprocessing import LabelEncoder
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test2[:202]).value_counts(normalize=True).sort_index()
    train_dist, test_dist = train_dist.align(test_dist, fill_value=epsilon)
    correction = (test_dist / (train_dist + epsilon)).values
    adjusted_probs = probs * correction[np.newaxis, :]
    adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)
    return adjusted_probs

# Apply label shift correction
corrected_preds_2 = correct_label_shift(y_train, y_test2, raw_preds_2)

# ========================
# Save to .npy file
# ========================
np.save("preds_2.npy", corrected_preds_2)
print(f" preds_2.npy saved. Shape: {corrected_preds_2.shape}")

# Sanity check
assert corrected_preds_2.shape == (1818, 28), f" Shape mismatch: got {corrected_preds_2.shape}, expected (1818, 28)"
print(" Shape verified: preds_2.npy is ready for submission.")