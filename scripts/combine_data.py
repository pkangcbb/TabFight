import pandas as pd

# --- MI data (tab separated .txt files) ---
col_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'label']

mi1_train = pd.read_csv('data/raw/edin_train_1.txt', sep='\t', header=None, names=col_names, engine='python')
mi1_test  = pd.read_csv('data/raw/edin_test_1.txt',  sep='\t', header=None, names=col_names, engine='python')
mi2_train = pd.read_csv('data/raw/edin_train_2.txt', sep='\t', header=None, names=col_names, engine='python')
mi2_test  = pd.read_csv('data/raw/edin_test_2.txt',  sep='\t', header=None, names=col_names, engine='python')

print("=== Individual file sizes ===")
print(f"mi1_train: {mi1_train.shape}")
print(f"mi1_test:  {mi1_test.shape}")
print(f"mi2_train: {mi2_train.shape}")
print(f"mi2_test:  {mi2_test.shape}")

mi = pd.concat([mi1_train, mi1_test, mi2_train, mi2_test], ignore_index=True)

print(f"MI combined: {mi.shape}")
print(mi['label'].value_counts())
print(f"\n=== MI first 5 rows ===")
print(mi.head())

mi.to_csv('data/combined/edin_mi_data.csv', index=False)

# --- Breast cancer ---
bc_train = pd.read_csv('data/raw/wdbc_train.csv')
bc_test  = pd.read_csv('data/raw/wdbc_test.csv')

print("\n=== Individual file sizes ===")
print(f"bc_train: {bc_train.shape}")
print(f"bc_test:  {bc_test.shape}")

bc = pd.concat([bc_train, bc_test], ignore_index=True)

print(f"\nBreastCancer combined: {bc.shape}")
print(bc['label'].value_counts())
print(f"\n=== BreastCancer first 5 rows ===")
print(bc.head())

bc.to_csv('data/combined/wdbc_data.csv', index=False)