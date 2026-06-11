# data_loader.py
# ─────────────────────────────────────────
# Single point of entry for data
# Load, preprocess, and subsample datasets.
# ─────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# get the root of the project regardless of where script is run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
load_dataset

PURPOSE: Insert name of dataset and extract features (X) and labels (y)
PARAMS : 
    name - string
RETURNS: 
    X (features) - pd.DataFrame
    y (labels)   - pd.Series
"""
def load_dataset(name):
    paths = {
        'mi': os.path.join(ROOT, 'data/combined/edin_mi_data.csv'),
        'bc': os.path.join(ROOT, 'data/combined/wdbc_data.csv')
    }

    if name not in paths: 
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(paths.keys())}")
    
    df = pd.read_csv(paths[name])

    X = df.drop(columns=['label'])
    y = df['label']

    print(f"Loaded {name}: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts().to_dict()}")
    print(f"Positive rate: {y.mean():.2%}\n")

    return X, y


"""
preprocess

PURPOSE: impute (fill in missing values) and encode (convert categorical features to numeric)
    *NOTE*: imputation (median calculation and most frequent selection) is based on training data only to avoid data leakage (model already "seeing" test data)
PARAMS: 
    X_train - pd.DataFrame
    X_test  - pd.DataFrame
RETURNS:
    X_train_processed - pd.DataFrame
    X_test_processed  - pd.DataFrame
"""
def preprocess(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # --- impute numeric column --- #
    if num_cols:
        imputer = SimpleImputer(strategy='median')
        X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = imputer.transform(X_test[num_cols])

    # --- impute and encode categorical columns --- #
    if cat_cols:
        # impute with most frequent category
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols]  = cat_imputer.transform(X_test[cat_cols])

        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_test[cat_cols]  = encoder.transform(X_test[cat_cols])
    
    return X_train, X_test

"""
subsample

PURPOSE: stratify a subsample that preverse the original class balance (positive rate)e
PARAMS:
    X - pd.DataFrame
    y - pd.Series
    n - int (number of samples in subsample)
    random_state - int (for reproducibility)
RETURNS:
    X_subsample - pd.DataFrame
    y_subsample - pd.Series
"""
def subsample(X_train, y_train, n, random_state=None):
    if n > len(X_train):
        raise ValueError(f"Subsample size n={n} cannot be larger than dataset size {len(X_train)}")
    
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n,
        random_state=random_state
    )
    idx, _ = next(sss.split(X_train, y_train))

    return X_train.iloc[idx].reset_index(drop=True), \
           y_train.iloc[idx].reset_index(drop=True)