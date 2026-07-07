# config.py
# ─────────────────────────────────────────
# All experiment parameters in one place.
# Change things here, not buried in code.
# ─────────────────────────────────────────

# Paths
DATA_DIR        = 'data/combined/'
RESULTS_DIR     = 'results/raw/'
FIGURES_DIR     = 'results/figures/'

# Datasets
DATASETS = {
    'MI':           'data/combined/edin_mi_data.csv',
    'BreastCancer': 'data/combined/wdbc_data.csv',
}
TARGET_COL = 'label'

# Cross-validation
N_SPLITS        = 5
N_REPEATS       = 10
RANDOM_STATE    = 42

# Training size ablation
TRAIN_SIZES     = [50, 100, 200, 500, None]  # None = full training set

# Hyperparameter tuning
N_SEARCH_ITER   = 50   # random search configurations per model
N_INNER_SPLITS  = 3    # inner CV folds for tuning

# Hardware
DEVICE = 'cuda' 