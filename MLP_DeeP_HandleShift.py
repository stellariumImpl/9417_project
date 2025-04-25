import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Weighted log loss (for label shift evaluation)
def weighted_log_loss(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    n_classes = y_pred.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]
    class_counts = y_true_onehot.sum(axis=0)
    present_mask = class_counts > 0
    class_weights = np.zeros_like(class_counts)
    class_weights[present_mask] = 1.0 / (class_counts[present_mask] + 1e-6)
    class_weights /= class_weights.sum()
    sample_weights = (y_true_onehot * class_weights).sum(axis=1)
    log_probs = np.log(y_pred + 1e-8)
    loss_per_sample = -np.sum(y_true_onehot * log_probs, axis=1)
    loss = np.mean(sample_weights * loss_per_sample)
    return loss

# Set random seed for reproducibility
def seed_everything(seed=42):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# Load data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")["label"]
X_test2 = pd.read_csv("X_test_2.csv")
y_test2_labeled = pd.read_csv("y_test_2_reduced.csv")["label"]

# Split test2 into labeled and unlabeled subsets
X_test2_labeled = X_test2.iloc[:len(y_test2_labeled)].copy()
X_test2_unlabeled = X_test2.iloc[len(y_test2_labeled):].copy()

# Combine train + test2 labeled as complete training set
X_full = pd.concat([X_train, X_test2_labeled], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_test2_labeled], axis=0).reset_index(drop=True)

# Model architecture
class MLPDeep(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 28)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn(x)
        x = self.dropout(x)
        return self.out(x)

class WeightedDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.w = torch.tensor(weights, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.w[idx]

def weighted_ce_loss(logits, targets, weights):
    ce = F.cross_entropy(logits, targets, reduction='none')
    return (weights * ce).mean()

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_macro_f1, all_weighted_f1, all_ce, all_weighted_ce = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    print(f"\n===== Fold {fold + 1}/5 =====")
    X_tr = X_full.iloc[train_idx]
    y_tr = y_full.iloc[train_idx]
    X_val = X_full.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    # Covariate shift estimation using domain classifier
    X_domain = pd.concat([X_tr, X_test2_unlabeled], axis=0)
    y_domain = np.array([0] * len(X_tr) + [1] * len(X_test2_unlabeled))
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_domain, y_domain)
    train_prob = clf.predict_proba(X_tr)[:, 1]
    sample_weights = train_prob / (1 - train_prob + 1e-6)

    # Standardization + feature selection
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    selector = SelectKBest(score_func=mutual_info_classif, k=200)
    X_tr_selected = selector.fit_transform(X_tr_scaled, y_tr)
    X_val_selected = selector.transform(X_val_scaled)

    # SMOTE oversampling and propagate weights
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_final, y_train_final = smote.fit_resample(X_tr_selected, y_tr)
    index_array = np.arange(len(sample_weights)).reshape(-1, 1)
    resampled_indices, _ = smote.fit_resample(index_array, y_tr)
    sample_weights_final = sample_weights[resampled_indices.flatten()]

    # DataLoader
    train_dataset = WeightedDataset(X_train_final, y_train_final, sample_weights_final)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32)
    y_val_array = y_val.values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPDeep(input_dim=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Training loop
    for epoch in range(30):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/30", unit="batch")
        for xb, yb, wb in pbar:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = weighted_ce_loss(out, yb, wb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        scheduler.step(total_loss)

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor.to(device))
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_preds = np.argmax(val_probs, axis=1)

    macro_f1 = f1_score(y_val_array, val_preds, average='macro')
    weighted_f1 = f1_score(y_val_array, val_preds, average='weighted')
    ce = log_loss(y_val_array, val_probs, labels=list(range(28)))
    weighted_ce = weighted_log_loss(y_val_array, val_probs)

    all_macro_f1.append(macro_f1)
    all_weighted_f1.append(weighted_f1)
    all_ce.append(ce)
    all_weighted_ce.append(weighted_ce)

    print(f"Fold {fold+1} Results:")
    print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | CE: {ce:.4f} | WCE: {weighted_ce:.4f}")

# Summary of cross-validation results
print("\nCross-validation Summary")
print(f"Avg Macro F1:     {np.mean(all_macro_f1):.4f} ± {np.std(all_macro_f1):.4f}")
print(f"Avg Weighted F1:  {np.mean(all_weighted_f1):.4f} ± {np.std(all_weighted_f1):.4f}")
print(f"Avg CE Loss:      {np.mean(all_ce):.4f} ± {np.std(all_ce):.4f}")
print(f"Avg Weighted CE:  {np.mean(all_weighted_ce):.4f} ± {np.std(all_weighted_ce):.4f}")
