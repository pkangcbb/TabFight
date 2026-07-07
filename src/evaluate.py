import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import friedmanchisquare, wilcoxon

"""
compute_metrics

PURPOSE: Compute 3 metrics (roc_auc, average_precision, brier_score) for one model on one test fold
PARAMS:
    y_true - true binary labels (pd.Series or np.array of 0/1)
    y_prob - predicted probabilities for the positive class (pd.Series or np.array of floats)
RETURNS:
    metrics - dict of metric_name: score
"""
def compute_metrics(y_true, y_prob):
    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob)
    }


"""
    summarize_results

    PURPOSE: Aggregate raw per-fold results into mean/std per model x dataset x train_size
    PARAMS:
        results_df - DataFrame with columns ['dataset', 'model', 'train_size', 'auroc', 'auprc', 'brier', 'train_time', 'inf_time']
    RETURNS:
        summary_df - DataFrame with columns ['dataset', 'mode', 'train_size'] (one row per each one)
"""
def summarize_results(results_df):
    summary = (
        results_df
        .groupby(['dataset', 'model', 'train_size'])
        .agg(
            auroc_mean=('auroc', 'mean'),
            auroc_std=('auroc', 'std'),
            auprc_mean=('auprc', 'mean'),
            auprc_std=('auprc', 'std'),
            brier_mean=('brier', 'mean'),
            brier_std=('brier', 'std'),
            train_time_mean=('train_time', 'mean'),
            inf_time_mean=('inf_time', 'mean'),
            n_runs=('auroc', 'count')
        )
        .reset_index()
    )
    return summary

"""
    run_friedman_test
    
    PURPOSE: Test for significant differences in rankings across models for this dataset/train_size combination
    PARAMS:
        results_df - pd.DataFrame of raw per-fold results
        dataset_name - str
        train_size - value matching the 'train_size' column (e.g. None or 500)
        metric - str of which column to test ('auroc', 'auprc', 'brier')

    RETURNS:
        dict with keys - statistic, p_value, models (list)
"""
def run_friedman_test(results_df, dataset_name, train_size, metric='auroc'):

    df = results_df[
        (results_df['dataset'] == dataset_name) &
        (results_df['train_size'] == train_size)
    ]

    models = sorted(df['model'].unique())

    # collect metric values per model, aligned by fold
    pivot = df.pivot_table(index='fold', columns='model', values=metric)
    pivot = pivot.dropna()  # only folds where all models succeeded

    if pivot.shape[1] < 2:
        return {'statistic': None, 'p_value': None, 'models': models,
                'note': 'fewer than 2 models with complete data'}

    arrays = [pivot[m].values for m in pivot.columns]
    stat, p = friedmanchisquare(*arrays)

    return {
        'statistic': stat,
        'p_value': p,
        'models': list(pivot.columns),
        'n_folds': pivot.shape[0],
    }

"""
    run_pairwise_wilcoxon
    
    PURPOSE: signed-rank tests between all model pairs, with Bonferroni correction for multiple comparisons.
    PARAMS:
        results_df - pd.DataFrame of raw per-fold results
        dataset_name - str
        train_size - value matching the 'train_size' column
        metric - str
        alpha - float of significance level before correction
    RETURNS:
        pd.DataFrame with columns: model_a, model_b, statistic, p_value, significant
"""
def run_pairwise_wilcoxon(results_df, dataset_name, train_size, metric='auroc', alpha=0.05):
    df = results_df[
        (results_df['dataset'] == dataset_name) &
        (results_df['train_size'] == train_size)
    ]

    pivot = df.pivot_table(index='fold', columns='model', values=metric).dropna()
    models = list(pivot.columns)

    n_pairs = len(models) * (len(models) - 1) / 2
    alpha_corrected = alpha / n_pairs if n_pairs > 0 else alpha

    rows = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            stat, p = wilcoxon(pivot[m1], pivot[m2])
            rows.append({
                'model_a': m1,
                'model_b': m2,
                'statistic': stat,
                'p_value': p,
                'significant': p < alpha_corrected,
            })

    return pd.DataFrame(rows)
