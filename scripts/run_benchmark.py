"""
TabFight — main benchmark loop.
Runs all models on all datasets at all training sizes,
saves results to results/raw/results.csv
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

# make sure src/ is importable from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASETS, TARGET_COL, N_SPLITS, N_REPEATS,
    RANDOM_STATE, TRAIN_SIZES, RESULTS_DIR
)
from src.data_loader import load_dataset, preprocess, subsample
from src.models import get_all_models
from src.evaluate import compute_metrics


def run_benchmark():

    # storage for all results
    results = []

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")

        # load full dataset
        X, y = load_dataset(dataset_name)

        # outer CV: 5 folds x 10 repeats = 50 evaluation runs
        outer_cv = RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):

            print(f"\n  Fold {fold_idx+1}/50")

            # split into train and test
            X_train_full = X.iloc[train_idx].reset_index(drop=True)
            X_test       = X.iloc[test_idx].reset_index(drop=True)
            y_train_full = y.iloc[train_idx].reset_index(drop=True)
            y_test       = y.iloc[test_idx].reset_index(drop=True)

            # preprocess — fit on train only
            X_train_full, X_test = preprocess(X_train_full, X_test)

            for train_size in TRAIN_SIZES:

                # subsample training set if needed
                X_train, y_train = subsample(
                    X_train_full, y_train_full,
                    n=train_size if train_size is not None else len(X_train_full),
                    random_state=RANDOM_STATE + fold_idx  # different seed per fold
                )

                actual_train_size = len(X_train)

                # get fresh model instances for this fold
                models = get_all_models()

                for model_name, model in models.items():

                    try:
                        # --- train ---
                        t_start = time.perf_counter()

                        if model_name in ['XGBoost']:
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_test, y_test)],
                                verbose=False,
                            )
                        else:
                            model.fit(X_train, y_train)

                        t_train = time.perf_counter() - t_start

                        # --- predict ---
                        t_inf_start = time.perf_counter()
                        y_prob = model.predict_proba(X_test)[:, 1]
                        t_inf = (time.perf_counter() - t_inf_start) / len(X_test)

                        # --- metrics ---
                        metrics = compute_metrics(y_test, y_prob)

                        results.append({
                            'dataset':    dataset_name,
                            'fold':       fold_idx,
                            'train_size': actual_train_size,
                            'model':      model_name,
                            'auroc':      metrics['auroc'],
                            'auprc':      metrics['auprc'],
                            'brier':      metrics['brier'],
                            'train_time': t_train,
                            'inf_time':   t_inf,
                        })

                        print(f"    [{model_name}] n={actual_train_size} "
                              f"AUROC={metrics['auroc']:.3f} "
                              f"t={t_train:.2f}s")

                    except Exception as e:
                        print(f"    [{model_name}] FAILED: {e}")
                        results.append({
                            'dataset':    dataset_name,
                            'fold':       fold_idx,
                            'train_size': actual_train_size,
                            'model':      model_name,
                            'auroc':      np.nan,
                            'auprc':      np.nan,
                            'brier':      np.nan,
                            'train_time': np.nan,
                            'inf_time':   np.nan,
                        })

                    # save after every model run
                    # so if it crashes midway you don't lose everything
                    results_df = pd.DataFrame(results)
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    results_df.to_csv(
                        os.path.join(RESULTS_DIR, 'results.csv'),
                        index=False
                    )

    print(f"\n{'='*60}")
    print(f"DONE. Results saved to {RESULTS_DIR}results.csv")
    print(f"Total runs: {len(results)}")
    return pd.DataFrame(results)


if __name__ == '__main__':
    results_df = run_benchmark()