import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import os
import random


def weighted_log_loss(y_true, y_pred):
    n_classes = y_pred.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]  # shape: (n_samples, n_classes)

    class_counts = np.sum(y_true_onehot, axis=0)  # 每类有多少样本
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true_onehot * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true_onehot * np.log(y_pred + 1e-8), axis=1))
    return loss


# ========== 固定随机种子 ==========
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ========== 自定义 Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


# ========== 模型结构 ==========
class MLPDeep(nn.Module):
    def __init__(self, input_dim):
        super(MLPDeep, self).__init__()
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


# ========== 主 K 折训练流程 ==========
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")["label"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_f1_macro = []
all_f1_weighted = []
all_loss = []
all_weighted_loss = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold + 1} =====")

    # 划分当前折
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 特征选择
    k = 200
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)

    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)

    # 转 tensor
    X_train_tensor = torch.tensor(X_train_smote, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_smote.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

    # 模型 & 优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPDeep(input_dim=k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = FocalLoss(alpha=1.0, gamma=1.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    best_model = None

    # 训练
    for epoch in range(30):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1}/30", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor.to(device))
            val_pred_labels = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_pred_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        f1_macro = f1_score(y_val, val_pred_labels, average='macro')
        f1_weighted = f1_score(y_val, val_pred_labels, average='weighted')
        ce = log_loss(y_val, val_pred_probs)
        weighted_loss = weighted_log_loss(y_val, val_pred_probs)
        print(f"Val Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f} | Loss: {ce:.4f} | Weighted_loss: {weighted_loss:.4f}")

        scheduler.step(ce)
        if ce < best_loss:
            best_loss = ce
            best_model = model.state_dict()

    all_f1_macro.append(f1_macro)
    all_f1_weighted.append(f1_weighted)
    all_loss.append(best_loss)
    all_weighted_loss.append(weighted_loss)

print("\n==== 最终交叉验证结果 ====")
print(f"Avg Macro F1:     {np.mean(all_f1_macro):.4f} | Std: {np.std(all_f1_macro):.4f}")
print(f"Avg Weighted F1:  {np.mean(all_f1_weighted):.4f} | Std: {np.std(all_f1_weighted):.4f}")
print(f"Avg CrossEntropy: {np.mean(all_loss):.4f} | Std: {np.std(all_loss):.4f}")
print(f"Avg Weighted CrossEntropy: {np.mean(all_weighted_loss):.4f}")
