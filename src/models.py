"""
Model registry for TabFight.
Each function returns a fresh, untrained model instance.
All models follow the sklearn API: .fit(X, y) and .predict_proba(X)
"""

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, DEVICE


def get_gbdt_models():
    """
    GBDT baselines — these are fast and run on CPU.
    Start here to validate your pipeline before adding heavier models.
    """
    models = {}

    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='auc',
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    models['CatBoost'] = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    models['RandomForest'] = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return models


"""
get _all_models

PURPOSE: Returns the full model registry as {name: model_instance}.
"""
def get_all_models():
    models = {}
    models.update(get_gbdt_models())
    return models