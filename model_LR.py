# Explore which features are most relevant to the classification of the predicted target
import pandas
import numpy as np
from itertools import combinations
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
# Model Training:
# print(X_smote)
# print(y_smote.value_counts())
# LogisticRegression model
# Learned from; https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
model_LR_new = LogisticRegression(
    penalty='l2',
    solver='newton-cg',
    multi_class='multinomial',
    max_iter=800,
    random_state=42,
    class_weight='balanced'

)
# training
model_LR_new.fit(X_smote,y_smote)

# print(type(X_smote))
# print(type(y_smote))
#### （1）LogisticRegression
y_val_pred_LR_new=model_LR_new.predict(X_val)

# to get log loss we need to use predict_proba
y_val_prob_LR_new = model_LR_new.predict_proba(X_val)

# Learned from: https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html
from sklearn.metrics import classification_report
# sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None)
print(classification_report(y_val, y_val_pred_LR_new))

# Learned from: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
from sklearn.metrics import f1_score
f1_macro_LR_new = f1_score(y_val, y_val_pred_LR_new, average='macro')
print(f"F1-Score Macro (Logistic Regression): {f1_macro_LR_new}")
f1_weighted_LR_new = f1_score(y_val, y_val_pred_LR_new, average='weighted')
print(f"F1-Score Weighted (Logistic Regression): {f1_weighted_LR_new}")

# Learned from:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
# sklearn.metrics.log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None)
# We need to use prob here NOT pred
from sklearn.metrics import log_loss
LR_logloss_new = log_loss(y_val, y_val_prob_LR_new)
print(f"Log Loss(cross-entropy)(Logistic Regression): {LR_logloss_new}")

# Learned from:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
from sklearn.metrics import confusion_matrix
cm_new = confusion_matrix(y_val, y_val_pred_LR_new, labels=range(28))
print("Confusion Matrix:")
print(cm_new)

# pred test dataset
import numpy as np

X_test_1_selectedFeatures = X_test_1_df[top250_features]
preds_1 = model_LR_new.predict_proba(X_test_1_selectedFeatures)
np.save("preds_1.npy", preds_1)
print("preds_1.npy file saved")