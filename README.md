### Project Structure

This project is aimed at process training set and test set distribution shift including Covariate shift, Label shift, Concept shift in multiclass classification tasks using various machine learning models, for using Logistic Regression as baseline, Random Forest, XGBoost, cost-sensitive XGBoost, and with DANN and Deep MLP. The content below also provides modular scripts for EDA, discuss different methods of feature selection, shift handling, and model training.

```
code/
│
├── Cost-SensitiveXGBoost.py          # XGBoost model with cost-sensitive learning (handles class imbalance)
├── DANN_XGBoost_HandleShift.py       # Domain-Adversarial Neural Network (DANN) + XGBoost for shift handling
├── distribution_shift.py             # Core functions for shift detection and mitigation (covariate, label, concept)
├── eda.py                            # Exploratory Data Analysis (EDA) script
├── feature_selection_default.py      # Baseline feature selection script
├── LogisticRegression.py             # Standard Logistic Regression model training
├── LogisticRegression_HandleShift.py # Logistic Regression with shift handling strategies
├── MLP_Deep.py                        # Deep Multi-Layer Perceptron (MLP) model
├── MLP_Deep_HandleShift.py            # MLP model with shift handling (e.g., reweighting, adaptation)
├── RF.py                              # Random Forest (RF) baseline model
├── RF_HandleShift.py                  # RF with shift mitigation techniques
├── utils.py                           # Utility functions (e.g., metrics, data processing)
├── Xgb.py                             # XGBoost baseline model
└── Xgb_HandleShift.py                 # XGBoost with shift correction methods

```

### Set Up and Run this Project

```bash
conda create -n py310 python=3.10
conda activate py310

# Dependencies via Conda
conda install numpy pandas scipy scikit-learn matplotlib seaborn lightgbm xgboost numba llvmlite statsmodels tqdm

# Remaining Dependencies via Pip
pip install -r requirements.txt
```

### Run the Scripts

- Perform **Exploratory Data Analysis (EDA)**:

```bash
python eda.py
```

- Feature Selection (Default Method)

```bash 
python feature_selection_default.py
# Baseline feature selection using statistical methods (e.g., Mutual Information, Fisher Score)
# Saves selected feature rankings for model training
```

- Detect **distribution shifts** (covariate shift, label shift, concept drift):

```bash
python distribution_shift.py
```

- Train **baseline models**:

```bash
python LogisticRegression.py
python RF.py
python Xgb.py
python MLP_Deep.py
```

- Train **models with shift mitigation**:

```bash
python LogisticRegression_HandleShift.py
python RF_HandleShift.py
python Xgb_HandleShift.py
python MLP_Deep_HandleShift.py
```

