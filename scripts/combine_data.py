import pandas as pd

# --- MI data (tab separated .txt files) ---
col_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'label']

mi1_train = pd.read_csv('data/raw/edin_train_1.txt', header=None, names=col_names)
mi1_test  = pd.read_csv('data/raw/edin_test_1.txt',  header=None, names=col_names)
mi2_train = pd.read_csv('data/raw/edin_train_2.txt', header=None, names=col_names)
mi2_test  = pd.read_csv('data/raw/edin_test_2.txt',  header=None, names=col_names)

mi = pd.concat([mi1_train, mi1_test, mi2_train, mi2_test], ignore_index=True)

# check for duplicates introduced by combining
mi = mi.drop_duplicates()

print(f"MI combined: {mi.shape}")

mi.to_csv('data/combined/edin_mi_data.csv', index=False)

# --- Breast cancer ---
bc_train = pd.read_csv('data/raw/wdbc_train.csv')
bc_test  = pd.read_csv('data/raw/wdbc_test.csv')

bc = pd.concat([bc_train, bc_test], ignore_index=True)
bc = bc.drop_duplicates()

print(f"\nBreastCancer combined: {bc.shape}")

bc.to_csv('data/combined/wdbc_data.csv', index=False)