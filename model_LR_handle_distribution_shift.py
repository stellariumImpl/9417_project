# Explore which features are most relevant to the classification of the predicted target
import pandas
import numpy as np
from itertools import combinations

# Learned from group work pdf
def weighted_log_loss(y_true, y_pred, epsilon=1e-15):
    """
    Compute the weighted log loss with class-based sample weights.

    Parameters:
    - y_true: (N, C) One-hot encoded true labels.
    - y_pred: (N, C) Predicted probabilities.
    - epsilon: Small value to avoid log(0).

    Returns:
    - loss (float): Weighted log loss.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # avoid log(0)

    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))

    return loss

# Now after we read this csv to dataframe, we can do some Feature Engineering on it
X_train=pandas.read_csv('./Data/X_train.csv')
# print(X_train)
y_train=pandas.read_csv('./Data/y_train.csv')
# print(y_train)

# First, check whether there are any missing values in X_train and y_train
# print(X_train.isnull().sum().sum())
# print(y_train.isnull().sum().sum())

# It was found that there were no missing values in the training set. Therefore, we do not need to handle the missing values in the training set
# So next, let's check the corresponding situation of the test set
X_test_1_df=pandas.read_csv('./Data/X_test_1.csv')
# print(X_test_1_df)

X_test_2_df=pandas.read_csv('./Data/X_test_2.csv')
# print(X_test_2_df)

y_test_2_reduced_df=pandas.read_csv('./Data/y_test_2_reduced.csv')
# print(y_test_2_reduced_df)

# Check the missing values of all test sets
# print(X_test_1_df.isnull().sum().sum())
# print(X_test_2_df.isnull().sum().sum())
# print(y_test_2_reduced_df.isnull().sum().sum())

# There are no missing values, so there is no need to handle the missing values. Next, directly calculate the average value of each feature in each class in the training set
# Merging the two DFS facilitates our statistics of the average value of a certain feature in a specific category
# Learned from: https://pandas.pydata.org/docs/reference/api/pandas.concat.html
# we can use pandas.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None)
Xy_train_df = pandas.concat([X_train, y_train], axis=1)
# print(Xy_train_df)

# Obtain the mean value of each feature in each category
# Learned from: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# we can use DataFrame.groupby(by=None, axis=<no_default>, level=None, as_index=True, sort=True, group_keys=True, observed=<no_default>, dropna=True)
class_feature_mean = Xy_train_df.groupby("label").mean()
# print(class_feature_mean)

# print(type(class_feature_mean))

import pandas as pd
feature_diffs = {}
for feature in class_feature_mean.columns:
    class_means = class_feature_mean[feature]
    diffs = [
        abs(class_means[i] - class_means[j])
        for i, j in combinations(class_means.index, 2)
    ]
    feature_diffs[feature] = np.mean(diffs)

sorted_features = pd.Series(feature_diffs).sort_values(ascending=False)
top250_features = sorted_features.head(250).index.tolist()
# print(top250_features)

Xy_train_mean = Xy_train_df[top250_features + ['label']]

# print(Xy_train_mean)

# Check the distribution of the categories in y_train and find that the category distribution is very unbalanced

# SMOTE
y_train_mean=Xy_train_mean['label']

X_train_mean=Xy_train_mean[top250_features]
from sklearn.model_selection import train_test_split
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_mean, y_train_mean, test_size=0.2, stratify=y_train_mean, random_state=42
    )

# print(X_subtrain)
# print(y_subtrain)

# First, test the number of samples for each category
# print(y_subtrain.value_counts())

# Learned from: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
from imblearn.over_sampling import SMOTE
# Because there are only 5 in category 16, k_neighbors should be less than 5
sm = SMOTE(random_state=42,k_neighbors=3)
X_smote, y_smote=sm.fit_resample(X_subtrain, y_subtrain)

############################################
# So there are many features' pvalues are small, so these 2 datasets have Covariate Shift
# handle Covariate Shift
# Apply the distribution of data from the new test set 2 to the training set
# Learned from paper：Learning and evaluating classifiers under sample selection bias
# Paper: https://dl.acm.org/doi/abs/10.1145/1015330.1015425
# In this paper, we need to train a classifier to classify which set the data belongs to after merging the original training set and the new test set
# class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
from sklearn.linear_model import LogisticRegression
dataset_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=100,
    random_state=42
)

# Merge the X of the training set and the test set
# Because we add one col named "dom
# So, we need to copy the original df to avoid changing in original df
X_train_labeled = X_train.copy()
X_train_labeled["dom"] = 0
X_test_labeled = X_test_2_df.copy()
X_test_labeled["dom"] = 1
# Learned from: https://pandas.pydata.org/docs/reference/api/pandas.concat.html
# pandas.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None)
Xdom_all_df = pd.concat([X_train_labeled, X_test_labeled]).reset_index(drop=True)

# get the mark of data source (Xtrain is 0, test2 is 1), for val
dom_col = np.array([0]*len(X_train) + [1]*202)  # 10000+202 = 10202

# Then we need to get X and y
X_all_addTest2 = Xdom_all_df.drop(columns="dom").to_numpy()
y_dom = Xdom_all_df["dom"].to_numpy()

# Training a dom classify model
dataset_clf.fit(X_all_addTest2, y_dom)
# get the prob of data in Xtrain in "X_train"
xTrain_p = dataset_clf.predict_proba(X_train)[:, 0]
# So the prob in Xtest2 would be 1-xTrain_p (because we only have 2 domain)
xTest_p=1-xTrain_p
# Learned from paper:https://dl.acm.org/doi/abs/10.1145/1015330.1015425
# we can allocate new weight to X_train's data
NewXTrainWeights=xTest_p/(xTrain_p+ 1e-5)

# Learned from: https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html#handling-data-shifts
# "The third approach is what is usually done in the industry today: retrain your model using the labeled data from the target distribution."
# but we need to use different weights
final_weight = np.concatenate([
    NewXTrainWeights * 1.0,         # allocate new weight to X_train's data
    np.full(202, 2.0)                # give more weight to the data that we already know in test2
])

##########################
from sklearn.linear_model import LogisticRegression

# combine data
X_test_2_labeled_df=X_test_2_df[:202]
# Learned from: https://pandas.pydata.org/docs/reference/api/pandas.concat.html
# pandas.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None)
X_all_df = pd.concat([X_train, X_test_2_labeled_df], axis=0).reset_index(drop=True)
y_all = np.concatenate([y_train.values.ravel(), y_test_2_reduced_df.values.ravel()])
X_all = X_all_df.values

###########################
# To get best parameters, we need to splite val dataset, and we also need to split our weight
X_train_HP, X_val_HP, y_train_final_HP, y_val_HP, dom_col_train_HP, dom_col_val_HP, final_weight_train_HP, final_weight_val_HP  = train_test_split(
    X_all, y_all, dom_col, final_weight,test_size=0.2, random_state=42, stratify=y_all
)

model_LR_shift_HP = LogisticRegression(
    penalty='l2', # Use l2 regularization to prevent overfitting, and under newton-cg, only l2 or no penalty term can be used
    C=1.0, # The reciprocal of the regularization strength, the smaller it is, the greater the weight given to the regularization penalty term
    solver='newton-cg', # We choose newton-cg as our optimization algorithm, which is suitable for multi-classification tasks and is a second-order optimization algorithm
    multi_class='multinomial', # "For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary."
    max_iter=800, # The maximum number of iterations is set to 800
    random_state=42, # Immediately, seed 42 was sown to facilitate the reproduction of the experimental results
    class_weight='balanced' # The category weights will be automatically adjusted according to the number of categories of the training samples to prevent poor training results in other categories when too many samples are concentrated in one category
)
# training
model_LR_shift_HP.fit(X_train_HP,y_train_final_HP,sample_weight=final_weight_train_HP)

#### get val metrics
# print(type(X_smote))
# print(type(y_smote))
#### （1）LogisticRegression
y_val_pred_HP=model_LR_shift_HP.predict(X_val_HP)

# to get log loss we need to use predict_proba
y_val_prob_HP = model_LR_shift_HP.predict_proba(X_val_HP)

# Learned from: https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html
from sklearn.metrics import classification_report
# sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None)
print(classification_report(y_val_HP, y_val_pred_HP))

# Learned from: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
from sklearn.metrics import f1_score
f1_macro_LR_HP = f1_score(y_val_HP, y_val_pred_HP, average='macro')
print(f"F1-Score Macro (Logistic Regression): {f1_macro_LR_HP}")
f1_weighted_LR_HP = f1_score(y_val_HP, y_val_pred_HP, average='weighted')
print(f"F1-Score Weighted (Logistic Regression): {f1_weighted_LR_HP}")

# Learned from:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
# sklearn.metrics.log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None)
# We need to use prob here NOT pred
from sklearn.metrics import log_loss
LR_logloss_HP = log_loss(y_val_HP, y_val_prob_HP)
print(f"Log Loss(cross-entropy)(Logistic Regression): {LR_logloss_HP}")

### weighted loss
# Learned from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html
# sklearn.preprocessing.label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False)
from sklearn.preprocessing import label_binarize
# One-hot encode y_val for all possible classes
y_ohe = label_binarize(y_val_HP, classes=np.arange(28))
valid_classes = np.unique(y_val_HP)
# Extract valid classes for weighted log loss
y_ohe_valid = y_ohe[:, valid_classes]
y_proba_valid = y_val_prob_HP[:, valid_classes]
wll = weighted_log_loss(y_ohe_valid, y_proba_valid)
print(f"Weighted Log Loss: {wll}")

###confusion matrix
# Learned from:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
from sklearn.metrics import confusion_matrix
cm_HP = confusion_matrix(y_val_HP, y_val_pred_HP, labels=range(28))
print("Confusion Matrix:")
print(cm_HP)


##########################
# Re Training Logistic Regression Model
model_LR_shift = LogisticRegression(
    penalty='l2', # Use l2 regularization to prevent overfitting, and under newton-cg, only l2 or no penalty term can be used
    C=1.0, # The reciprocal of the regularization strength, the smaller it is, the greater the weight given to the regularization penalty term
    solver='newton-cg', # We choose newton-cg as our optimization algorithm, which is suitable for multi-classification tasks and is a second-order optimization algorithm
    multi_class='multinomial', # "For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary."
    max_iter=800, # The maximum number of iterations is set to 800
    random_state=42, # Immediately, seed 42 was sown to facilitate the reproduction of the experimental results
    class_weight='balanced' # The category weights will be automatically adjusted according to the number of categories of the training samples to prevent poor training results in other categories when too many samples are concentrated in one category
)

# training all test2 labeled
model_LR_shift.fit(X_all,y_all,sample_weight=final_weight)
#########################
X_test_2_1818 = X_test_2_df.iloc[202:202+1818]
preds_2 = model_LR_shift.predict_proba(X_test_2_1818)
np.save("preds_2.npy", preds_2)
print("preds_2.npy file saved")
