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


def get_tfm_models():
    """
    Tabular Foundation Models — zero shot, no tuning needed.
    """
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier

    models = {}

    models['TabPFN'] = TabPFNClassifier(
        device=DEVICE,
        random_state=RANDOM_STATE,
    )

    models['TabICL'] = TabICLClassifier(
        device=DEVICE,
        random_state=RANDOM_STATE,
    )

    return models


def get_dl_models():
    """
    Deep learning models via PyTorch Tabular.
    These need feature column names passed at fit time —
    handled in run_benchmark.py.
    """
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import (
        TabNetModelConfig,
        TabTransformerConfig,
        FTTransformerConfig,
        GANDALFConfig,
    )
    from pytorch_tabular.config import (
        DataConfig,
        TrainerConfig,
        OptimizerConfig,
    )

    models = {}

    models['TabNet']         = 'TabNet'
    models['TabTransformer'] = 'TabTransformer'
    models['FTTransformer']  = 'FTTransformer'
    models['GANDALF']        = 'GANDALF'

    return models

def get_linear_models():
    from sklearn.linear_model import LogisticRegression
    
    models = {}
    
    models['LogisticRegression'] = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    return models

def get_all_models():
    """
    Returns the full model registry.
    Call get_gbdt_models() alone to test pipeline quickly.
    """
    models = {}
    models.update(get_gbdt_models())
    models.update(get_linear_models())
    models.update(get_tfm_models())
    return models