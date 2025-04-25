import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelBinarizer
from sklearn.metrics import f1_score, log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

SAVE_PATH = "Xgb_result"
os.makedirs(SAVE_PATH, exist_ok=True)

def winsorize_iqr(df, lower=1.5, upper=1.5):
    """
    Apply IQR-based winsorization to cap outliers in each column.

    Args:
        df (pd.DataFrame): Input feature DataFrame.
        lower (float): Lower bound multiplier for IQR (default 1.5).
        upper (float): Upper bound multiplier for IQR (default 1.5).

    Returns:
        pd.DataFrame: Winsorized DataFrame.
    """
    """
    Cap outliers using the IQR (Interquartile Range) method.
    """
    df_capped = df.copy()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df_capped[col] = df[col].clip(Q1 - lower * IQR, Q3 + upper * IQR)
    return df_capped

def weighted_log_loss(y_true, y_pred_proba):
    """
    Calculate weighted log loss, adjusted for class imbalance.

    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred_proba (np.ndarray): Predicted probabilities.

    Returns:
        float: Weighted log loss value.
    """
    """
    Compute weighted log loss to address class imbalance in multi-class classification.
    """
    class_counts = np.sum(y_true, axis=0)       
    nonzero_classes = class_counts > 0          
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    class_weights[nonzero_classes] = 1.0 / (class_counts[nonzero_classes] + 1e-8)
    class_weights /= np.sum(class_weights)       
    sample_weights = np.sum(y_true * class_weights, axis=1)  
    sample_losses = -np.sum(y_true * np.log(y_pred_proba + 1e-8), axis=1)
    return np.mean(sample_weights * sample_losses)

def compute_ior(X, y):
    """
    Compute improved Odds Ratio (iOR) scores for all features.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.

    Returns:
        np.ndarray: Array of iOR scores for each feature.
    """
    """
    Compute improved Odds Ratio (iOR) for each feature based on class-wise separation.
    """
    y_arr = y.values.ravel()
    classes = np.unique(y_arr)
    scores = []
    for j in range(X.shape[1]):
        feat = X.iloc[:, j]
        score = 0.0
        for c in classes:
            pos, neg = feat[y_arr == c], feat[y_arr != c]
            mu1, mu0 = pos.mean(), neg.mean()
            sigma1, sigma0 = pos.std(), neg.std()
            sigma1, sigma0 = (sigma1 or 1), (sigma0 or 1)
            score += abs((mu1 - mu0) / (sigma1 + sigma0))
        scores.append(score)
    return np.array(scores)

def compute_fisher_score(X, y):
    """
    Compute Fisher score for all features.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target labels.

    Returns:
        np.ndarray: Array of Fisher scores for each feature.
    """
    """
    Compute Fisher Score for each feature, measuring the ratio of between-class variance to within-class variance.
    """
    X, y = np.asarray(X), np.asarray(y)
    overall_mean = X.mean(axis=0)
    classes = np.unique(y)
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        num, denom = 0.0, 0.0
        for c in classes:
            X_c = X[y == c, j]
            n_c = len(X_c)
            mu_c, var_c = X_c.mean(), X_c.var()
            num += n_c * (mu_c - overall_mean[j])**2
            denom += n_c * var_c
        scores[j] = num / (denom + 1e-8)
    return scores

def evaluate_feature_selection(name, X_train, y_train, X_val, y_val, selected_columns):
    """
    Train and evaluate an XGBoost model using selected features.

    Args:
        name (str): Identifier for the current feature selection method.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series or np.ndarray): Validation labels.
        selected_columns (list): List of selected feature names.

    Returns:
        None. Prints classification report and saves confusion matrix plot.
    """
    """
    Train and evaluate an XGBoost model using a given subset of features.
    Save classification report and confusion matrix to disk.
    """
    print("\n>>> Evaluating Feature Selection: {} <<<".format(name))
    X_tr_iqr = winsorize_iqr(X_train[selected_columns])
    X_val_iqr = winsorize_iqr(X_val[selected_columns])
    scaler = RobustScaler()
    scaler_fitted = scaler.fit(X_tr_iqr)
    X_tr_scaled = scaler_fitted.transform(X_tr_iqr)
    X_val_scaled = scaler_fitted.transform(X_val_iqr)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_train)
    model = XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss',
                          n_estimators=100, random_state=42, tree_method='hist', device='cuda')
    model.fit(X_tr_res, y_tr_res)
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)
    lb = LabelBinarizer().fit(range(28))
    y_val_onehot = lb.transform(y_val)
    if y_val_onehot.shape[1] == 1:
        y_val_onehot = np.hstack([1 - y_val_onehot, y_val_onehot])
    print(classification_report(y_val, y_pred, zero_division=0))
    print("F1 Macro: {:.4f}".format(f1_score(y_val, y_pred, average='macro')))
    print("F1 Weighted: {:.4f}".format(f1_score(y_val, y_pred, average='weighted')))
    print("Log Loss: {:.4f}".format(log_loss(y_val, y_proba)))
    print("Weighted Log Loss: {:.4f}".format(weighted_log_loss(y_val_onehot, y_proba)))
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: {}'.format(name))
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "cm_{}.png".format(name)))
    plt.close()

def tune_model(X, y, base_model, param_grids):
    """
    Perform multi-stage GridSearchCV to find the best hyperparameters.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        base_model (sklearn estimator): XGBoost model to be tuned.
        param_grids (list): List of dictionaries, each representing a stage of hyperparameter grid.

    Returns:
        tuple: (Best model with tuned parameters, Dictionary of best parameters).
    """
    """
    Tune model hyperparameters in multiple stages using GridSearchCV.
    Returns the best model and parameter set.
    """
    best_params = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, param_grid in enumerate(param_grids, 1):
        print("\n>>> Grid Search Stage {} <<<".format(i))
        model = base_model.set_params(**best_params)
        gs = GridSearchCV(estimator=model, param_grid=param_grid,
                          scoring='f1_weighted', cv=cv, refit=True,
                          verbose=2, n_jobs=-1)
        gs.fit(X, y)
        best_params.update(gs.best_params_)
        print("Best parameters in stage {}: {}".format(i, gs.best_params_))
    best_model = base_model.set_params(**best_params)
    best_model.fit(X, y) 
    return best_model, best_params


def train_with_shift_correction(
    X_train_sub: pd.DataFrame,
    y_train_sub: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_features: list,
    best_params: dict
) -> None:
    """
    Train a classification model using Fisher-selected features while correcting for covariate shift
    and label shift between training and validation datasets.

    Parameters:
    -----------
    X_train_sub : pd.DataFrame
        Training features before final selection.
    y_train_sub : pd.Series
        Corresponding training labels.
    X_val : pd.DataFrame
        Validation set features.
    y_val : pd.Series
        Validation set labels.
    top_features : list
        Selected top-K features by Fisher Score.
    best_params : dict
        Best hyperparameters from grid search.

    Returns:
    --------
    None. Saves trained model and confusion matrix plot to disk.
    """
    X_fisher = winsorize_iqr(X_train_sub[top_features])
    scaler = RobustScaler().fit(X_fisher)
    X_fisher_scaled = scaler.transform(X_fisher)
    X_val_final = winsorize_iqr(X_val[top_features])
    X_val_final_scaled = scaler.transform(X_val_final)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_fisher_res, y_fisher_res = smote.fit_resample(X_fisher_scaled, y_train_sub)

    X_domain = np.vstack([X_fisher_scaled, X_val_final_scaled])
    y_domain = np.array([0]*len(X_fisher_scaled) + [1]*len(X_val_final_scaled))
    domain_clf = LogisticRegression(max_iter=1000).fit(X_domain, y_domain)
    p_test = domain_clf.predict_proba(X_fisher_res)[:, 1]
    p_train = 1 - p_test
    importance_weights = p_test / (p_train + 1e-6)

    model_cov = XGBClassifier(**best_params)
    model_cov.fit(X_fisher_res, y_fisher_res, sample_weight=importance_weights)

    def estimate_confusion_matrix(model, X, y, num_classes):
        pred = model.predict(X)
        cm = confusion_matrix(y, pred, labels=range(num_classes))
        cm_n = cm / cm.sum(axis=1, keepdims=True)
        return np.nan_to_num(cm_n).T

    X_base, _, y_base, _ = train_test_split(X_fisher_res, y_fisher_res, test_size=0.8, stratify=y_fisher_res, random_state=42)
    C_T = estimate_confusion_matrix(model_cov, X_base, y_base, 28)
    avg_test_pred = model_cov.predict_proba(X_val_final_scaled).mean(axis=0)
    q_test = np.linalg.lstsq(C_T, avg_test_pred, rcond=None)[0]
    q_test = np.clip(q_test, 0, 1)
    q_test /= q_test.sum()
    train_prior = pd.Series(y_fisher_res).value_counts(normalize=True).sort_index().values

    def adjust_labels(model, X, test_priors, train_priors):
        proba = model.predict_proba(X)
        ratio = test_priors / (train_priors + 1e-8)
        adj = proba * ratio[np.newaxis, :]
        adj /= adj.sum(axis=1, keepdims=True)
        return np.argmax(adj, axis=1)

    y_adjusted = adjust_labels(model_cov, X_fisher_res, q_test, train_prior)

    model_final = XGBClassifier(**best_params)
    model_final.fit(X_fisher_res, y_adjusted, sample_weight=importance_weights)

    y_val_pred = model_final.predict(X_val_final_scaled)
    y_val_proba = model_final.predict_proba(X_val_final_scaled)

    lb = LabelBinarizer().fit(range(28))
    y_val_onehot = lb.transform(y_val)
    if y_val_onehot.shape[1] == 1:
        y_val_onehot = np.hstack([1 - y_val_onehot, y_val_onehot])

    print("\n=== Final Model Evaluation with Covariate & Label Shift ===")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print("Accuracy: {:.4f}".format(accuracy_score(y_val, y_val_pred)))
    print("F1 Macro: {:.4f}".format(f1_score(y_val, y_val_pred, average='macro')))
    print("F1 Weighted: {:.4f}".format(f1_score(y_val, y_val_pred, average='weighted')))
    print("Log Loss: {:.4f}".format(log_loss(y_val, y_val_proba)))
    print("Weighted Log Loss: {:.4f}".format(weighted_log_loss(y_val_onehot, y_val_proba)))

    cm_final = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Final Model (Shift Corrected)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "cm_final_model_2.png"))
    plt.close()

    model_final.save_model(os.path.join(SAVE_PATH, "final_model_2.json"))

def train_with_shift_correction(
    scaler: RobustScaler,
    X_ior_scaled: np.ndarray,
    X_ior_res: np.ndarray,
    y_ior_res: np.ndarray,
    top_features: list,
    best_params: dict
) -> None:
    """
    Train a classification model using Fisher-selected features while correcting for covariate shift
    and label shift between training and validation datasets.

    Parameters:
    -----------
    scaler : RobustScaler
        The fitted scaler used for normalizing input features during training.
    X_ior_scaled : np.ndarray
        Scaled and winsorized feature matrix before SMOTE, used for domain adaptation.
    X_ior_res : np.ndarray
        Resampled feature matrix after applying SMOTE to balance class distribution.
    y_ior_res : np.ndarray
        Corresponding labels for the SMOTE-resampled training set.
    top_features : list
        Selected top-K features by Fisher Score.
    best_params : dict
        Best hyperparameters from grid search.

    Returns:
    --------
    None. Saves trained model and confusion matrix plot to disk.
    """

    # === Extra step: Test on X_test_2 with covariate + label shift correction ===
    X_test_2 = pd.read_csv("data/X_test_2.csv").iloc[:202].copy()
    y_test_2_reduced = pd.read_csv("data/y_test_2_reduced.csv")[:202].squeeze()

    top_ior=top_features
    X_test_2_iqr = winsorize_iqr(X_test_2[top_ior])
    X_test_2_scaled = scaler.transform(X_test_2_iqr)
    X_domain = np.vstack([X_ior_scaled, X_test_2_scaled])
    y_domain = np.array([0]*len(X_ior_scaled) + [1]*len(X_test_2_scaled))
    domain_clf = LogisticRegression(max_iter=1000).fit(X_domain, y_domain)

    p_test = domain_clf.predict_proba(X_ior_res)[:, 1]
    p_train = 1 - p_test
    importance_weights = p_test / (p_train + 1e-6)

    model_cov = XGBClassifier(**best_params)
    model_cov.fit(X_ior_res, y_ior_res, sample_weight=importance_weights)

    def estimate_confusion_matrix(model, X, y, num_classes):
        pred = model.predict(X)
        cm = confusion_matrix(y, pred, labels=range(num_classes))
        cm_n = cm / cm.sum(axis=1, keepdims=True)
        return np.nan_to_num(cm_n).T

    X_base, _, y_base, _ = train_test_split(X_ior_res, y_ior_res, test_size=0.8, stratify=y_ior_res, random_state=42)
    C_T = estimate_confusion_matrix(model_cov, X_base, y_base, 28)
    avg_test_pred = model_cov.predict_proba(X_test_2_scaled).mean(axis=0)
    q_test = np.linalg.lstsq(C_T, avg_test_pred, rcond=None)[0]
    q_test = np.clip(q_test, 0, 1)
    q_test /= q_test.sum()
    train_prior = pd.Series(y_ior_res).value_counts(normalize=True).sort_index().values

    def adjust_labels(model, X, test_priors, train_priors):
        proba = model.predict_proba(X)
        ratio = test_priors / (train_priors + 1e-8)
        adj = proba * ratio[np.newaxis, :]
        adj /= adj.sum(axis=1, keepdims=True)
        return np.argmax(adj, axis=1)

    y_test_adjusted = adjust_labels(model_cov, X_ior_res, q_test, train_prior)
    final_model_shifted = XGBClassifier(**best_params)
    final_model_shifted.fit(X_ior_res, y_test_adjusted, sample_weight=importance_weights)

    y_pred_test2 = final_model_shifted.predict(X_test_2_scaled)
    y_proba_test2 = final_model_shifted.predict_proba(X_test_2_scaled)

    lb = LabelBinarizer()
    lb.fit(range(28))  
    y_test_oh = lb.transform(y_test_2_reduced)

    if y_test_oh.shape[1] < 28:
        pad_width = 28 - y_test_oh.shape[1]
        padding = np.zeros((y_test_oh.shape[0], pad_width))
        y_test_oh = np.hstack([y_test_oh, padding])
    
    print("\n=== Performance on X_test_2 with Shift Correction ===")
    print(classification_report(y_test_2_reduced, y_pred_test2, zero_division=0))
    print("Accuracy: {:.4f}".format(accuracy_score(y_test_2_reduced, y_pred_test2)))
    print("F1 Macro: {:.4f}".format(f1_score(y_test_2_reduced, y_pred_test2, average='macro')))
    print("F1 Weighted: {:.4f}".format(f1_score(y_test_2_reduced, y_pred_test2, average='weighted')))
    print("Log Loss: {:.4f}".format(log_loss(y_test_2_reduced, y_proba_test2, labels=range(28))))
    print("Weighted Log Loss: {:.4f}".format(weighted_log_loss(y_test_oh, y_proba_test2)))

    cm_test = confusion_matrix(y_test_2_reduced, y_pred_test2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set with Shift Correction')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "Xgb_Corvariate_Label_Shift.png"))
    plt.close()


def predict_on_test_all(feature_selections):
    """
    Load the trained model and use it to predict all rows of X_test_1.
    Saves predictions as 'preds_1.npy'.
    """
    model_path = os.path.join(SAVE_PATH, "final_model_1.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found: final_model_1.json")

    # Load trained model
    model = XGBClassifier()
    model.load_model(model_path)

    # Load test data
    X_test_1 = pd.read_csv("data/X_test_1.csv")

    # Replace with your actual selected top fisher features
    top_ior = feature_selections

    # Apply the same winsorization and scaling used during training
    X_test_iqr = winsorize_iqr(X_test_1[top_ior])
    X_train = pd.read_csv("data/X_train.csv")
    X_train_iqr = winsorize_iqr(X_train[top_ior])
    scaler = RobustScaler().fit(X_train_iqr)
    X_test_scaled = scaler.transform(X_test_iqr)

    # Make predictions and save
    preds = model.predict(X_test_scaled)
    np.save(os.path.join(SAVE_PATH, "preds_1.npy"), preds)
    print("Saved predictions to preds_1.npy")

def main():
    X = pd.read_csv("data/X_train.csv")
    y = pd.read_csv("data/y_train.csv").squeeze()
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train_iqr = winsorize_iqr(X_train_sub)
    scaler_fitted = RobustScaler().fit(X_train_iqr)
    X_train_scaled = pd.DataFrame(scaler_fitted.transform(X_train_iqr), columns=X.columns)
    X_val_iqr = winsorize_iqr(X_val)
    X_val_scaled = pd.DataFrame(scaler_fitted.transform(X_val_iqr), columns=X.columns)

    ior_scores = compute_ior(X_train_scaled, y_train_sub)
    k_list = list(range(1, 301))
    results_ior = []
    for k in tqdm(k_list, desc="Evaluating iOR top-K"):
        cols_k = X.columns[np.argsort(ior_scores)[::-1][:k]]
        Xk_train = X_train_scaled[cols_k]
        Xk_val = X_val_scaled[cols_k]
        model = XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss',
                              n_estimators=100, random_state=42, tree_method='hist', device='cuda')
        model.fit(Xk_train, y_train_sub)
        y_pred = model.predict(Xk_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        results_ior.append((k, f1))
    best_k_ior, _ = max(results_ior, key=lambda x: x[1])
    print(f"The best k in IOR method is {best_k_ior}")
    ior_df = pd.DataFrame({
        'Feature': X.columns,
        'iOR_score': ior_scores
    }).sort_values(by='iOR_score', ascending=False).reset_index(drop=True)
    print("Top features by iOR score:")
    print(ior_df.head(best_k_ior))
    top_ior = ior_df.head(best_k_ior)['Feature'].tolist()
    evaluate_feature_selection("iOR", X_train_sub, y_train_sub, X_val, y_val, top_ior)

    results_ftest = []
    for k in tqdm(k_list, desc="Evaluating F-test top-K"):
        selector = SelectKBest(score_func=f_classif, k=k)
        Xk_train = selector.fit_transform(X_train_scaled, y_train_sub)
        Xk_val = selector.transform(X_val_scaled)
        model = XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss',
                              n_estimators=100, random_state=42, tree_method='hist', device='cuda')
        model.fit(Xk_train, y_train_sub)
        y_pred = model.predict(Xk_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        results_ftest.append((k, f1))
    best_k_ftest, _ = max(results_ftest, key=lambda x: x[1])
    print(f"The best k in F-test method is {best_k_ftest}")
    selector = SelectKBest(score_func=f_classif, k=best_k_ftest)
    selector.fit(X_train_scaled, y_train_sub)

    f_scores = selector.scores_
    p_values = selector.pvalues_
    feature_scores_df = pd.DataFrame({
        'Feature': X.columns,
        'F_score': f_scores,
        'p_value': p_values
    }).sort_values(by='F_score', ascending=False).reset_index(drop=True)
    print("Top features by F-test with statistical scores:")
    print(feature_scores_df.head(best_k_ftest))
    top_ftest = feature_scores_df.head(best_k_ftest)['Feature'].tolist()
    evaluate_feature_selection("F-test", X_train_sub, y_train_sub, X_val, y_val, top_ftest)

    fisher_scores = compute_fisher_score(X_train_scaled, y_train_sub)
    results_fisher = []
    for k in tqdm(k_list, desc="Evaluating Fisher top-K"):
        cols_k = X.columns[np.argsort(fisher_scores)[::-1][:k]]
        Xk_train = X_train_scaled[cols_k]
        Xk_val = X_val_scaled[cols_k]
        model = XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss',
                              n_estimators=100, random_state=42, tree_method='hist', device='cuda')
        model.fit(Xk_train, y_train_sub)
        y_pred = model.predict(Xk_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        results_fisher.append((k, f1))
    best_k_fisher, _ = max(results_fisher, key=lambda x: x[1])
    print(f"The best k in Fisher Score method is {best_k_fisher}")
    fisher_df = pd.DataFrame({
        'Feature': X.columns,
        'Fisher_score': fisher_scores
    }).sort_values(by='Fisher_score', ascending=False).reset_index(drop=True)
    print("Top features by Fisher Score:")
    print(fisher_df.head(best_k_fisher))
    top_fisher = fisher_df.head(best_k_fisher)['Feature'].tolist()
    evaluate_feature_selection("FisherScore", X_train_sub, y_train_sub, X_val, y_val, top_fisher)


    # Using Feature Selection IOR method to train and tune the model
    X_ior = winsorize_iqr(X_train_sub[top_ior])
    scaler = RobustScaler().fit(X_ior)
    X_ior_scaled = scaler.transform(X_ior)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_ior_res, y_ior_res = smote.fit_resample(X_ior_scaled, y_train_sub)

    base_model = XGBClassifier(objective='multi:softprob', num_class=28, eval_metric='mlogloss',
                               tree_method='hist', device='cuda', random_state=42)
    param_grids = [
        {'n_estimators': [100, 300, 500, 700, 1000], 'learning_rate': [0.01, 0.05, 0.1, 0.3]},
        {'max_depth': [3, 6, 10], 'min_child_weight': [1, 3, 5]},
        {'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'gamma': [0, 0.1, 0.3, 0.5]},
        {'reg_alpha': [0, 0.5, 1], 'reg_lambda': [1, 5, 10]}
    ]
    final_model, best_params = tune_model(X_ior_res, y_ior_res, base_model, param_grids)
    print(f"The best params in XGB model are: {best_params} ")
    
    X_val_final = winsorize_iqr(X_val[top_ior])
    X_val_final_scaled = scaler.transform(X_val_final)
    y_val_pred = final_model.predict(X_val_final_scaled)
    y_val_proba = final_model.predict_proba(X_val_final_scaled)
    lb = LabelBinarizer().fit(range(28))
    y_val_onehot = lb.transform(y_val)
    if y_val_onehot.shape[1] == 1:
        y_val_onehot = np.hstack([1 - y_val_onehot, y_val_onehot])

    print("\n=== Final Model Evaluation ===")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print("F1 Macro: {:.4f}".format(f1_score(y_val, y_val_pred, average='macro')))
    print("Accuracy: {:.4f}".format(accuracy_score(y_val, y_val_pred)))
    print("F1 Weighted: {:.4f}".format(f1_score(y_val, y_val_pred, average='weighted')))
    print("Log Loss: {:.4f}".format(log_loss(y_val, y_val_proba)))
    print("Weighted Log Loss: {:.4f}".format(weighted_log_loss(y_val_onehot, y_val_proba)))

    cm_final = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Final Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "cm_final_model.png"))
    plt.close()

    final_model.save_model(os.path.join(SAVE_PATH, "final_model_1.json"))

    train_with_shift_correction(scaler, X_ior_scaled, X_ior_res, y_ior_res, top_ior, best_params)

    predict_on_test_all(top_ior)

if __name__ == "__main__":
    main()
