"""
TabFight — ensemble benchmark.
Tests all 63 non-empty combinations of 6 models using 3 ensemble methods:
1. Simple average (mean of probabilities)
2. Weighted average (weight by individual AUROC)
3. Stacking (logistic regression meta-model)
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

def load_preds(path):
    df = pd.read_csv(path)
    df['y_prob'] = df['y_prob'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    df['y_true'] = df['y_true'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    return df

def simple_average(probs):
    return np.mean(probs, axis=0)

def weighted_average(probs, weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.average(probs, axis=0, weights=weights)

def stacking(probs, y_true):
    X_meta = np.column_stack(probs)
    if len(np.unique(y_true)) < 2:
        return simple_average(probs)
    try:
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(X_meta, y_true)
        return meta.predict_proba(X_meta)[:, 1]
    except Exception:
        return simple_average(probs)

def run_ensembles():

    pred_path = os.path.join(RESULTS_DIR, 'results_with_preds.csv')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found at {pred_path}.")

    df = load_preds(pred_path)
    
    # use only the 6 models for ensembling (exclude logistic regression)
    ensemble_models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'TabPFN', 'TabICL']
    df = df[df['model'].isin(ensemble_models)]
    models = sorted(df['model'].unique().tolist())
    
    print(f"Models: {models}")
    print(f"Total combinations: {2**len(models)-1}")

    ensemble_results = []

    for dataset in df['dataset'].unique():
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset}")

        df_ds = df[df['dataset'] == dataset]

        for train_size in sorted(df_ds['train_size'].unique()):
            df_ts = df_ds[df_ds['train_size'] == train_size]
            model_aurocs = df_ts.groupby('model')['auroc'].mean().to_dict()

            for r in range(1, len(models)+1):
                for combo in combinations(models, r):
                    combo_name = '+'.join(sorted(combo))
                    fold_results = {'simple': [], 'weighted': [], 'stacking': []}

                    for fold in df_ts['fold'].unique():
                        df_fold = df_ts[
                            (df_ts['fold'] == fold) &
                            (df_ts['model'].isin(combo))
                        ]

                        if len(df_fold) < len(combo):
                            continue

                        probs  = np.array(df_fold['y_prob'].tolist())
                        y_true = df_fold.iloc[0]['y_true']
                        weights = [model_aurocs.get(m, 0.5) for m in combo]

                        if len(combo) == 1:
                            prob_simple   = probs[0]
                            prob_weighted = probs[0]
                            prob_stack    = probs[0]
                        else:
                            prob_simple   = simple_average(probs)
                            prob_weighted = weighted_average(probs, weights)
                            prob_stack    = stacking(probs, y_true)

                        for method, prob in [
                            ('simple',   prob_simple),
                            ('weighted', prob_weighted),
                            ('stacking', prob_stack),
                        ]:
                            try:
                                auroc = roc_auc_score(y_true, prob)
                                auprc = average_precision_score(y_true, prob)
                                brier = brier_score_loss(y_true, prob)
                                fold_results[method].append({
                                    'auroc': auroc,
                                    'auprc': auprc,
                                    'brier': brier,
                                })
                            except Exception:
                                pass

                    for method, folds in fold_results.items():
                        if not folds:
                            continue
                        aurocs = [f['auroc'] for f in folds]
                        auprcs = [f['auprc'] for f in folds]
                        briers = [f['brier'] for f in folds]

                        ensemble_results.append({
                            'dataset':    dataset,
                            'train_size': train_size,
                            'combo':      combo_name,
                            'n_models':   len(combo),
                            'method':     method,
                            'auroc_mean': np.mean(aurocs),
                            'auroc_std':  np.std(aurocs),
                            'auprc_mean': np.mean(auprcs),
                            'brier_mean': np.mean(briers),
                        })

            print(f"  train_size={train_size} done")

    results_df = pd.DataFrame(ensemble_results)
    out_path = os.path.join(RESULTS_DIR, 'ensemble_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nDone. Saved to {out_path}")
    print(f"Total ensemble evaluations: {len(results_df)}")

    print("\n=== Top 10 ensembles on MI (full training, simple average) ===")
    mi = results_df[
        (results_df['dataset'] == 'MI') &
        (results_df['method'] == 'simple')
    ]
    mi_full_size = mi['train_size'].max()
    top = (mi[mi['train_size'] == mi_full_size]
           .sort_values('auroc_mean', ascending=False)
           .head(10))
    print(top[['combo', 'n_models', 'auroc_mean', 'auroc_std']].to_string(index=False))

    return results_df

if __name__ == '__main__':
    run_ensembles()