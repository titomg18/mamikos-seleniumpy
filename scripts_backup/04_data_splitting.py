import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_and_save(df, target_col, test_size, random_state, prefix, split_name):
    # Memisahkan fitur (X) dan target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Melakukan Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Menyimpan file dengan nama yang spesifik untuk rasionya
    out_dir = r'..\data\processed'
    
    X_train.to_csv(os.path.join(out_dir, f'X_train_{prefix}_{split_name}.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, f'X_test_{prefix}_{split_name}.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, f'y_train_{prefix}_{split_name}.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, f'y_test_{prefix}_{split_name}.csv'), index=False)
    
    return len(X_train), len(X_test)

def main():
    paths = {
        'drop': r'..\data\processed\03_encoded_drop.csv',
        'impute': r'..\data\processed\03_encoded_impute.csv'
    }
    
    print("=== PROSES SPLITTING DATA (MULTI-RATIO) ===")
    
    TARGET = 'harga'
    RANDOM_STATE = 42
    
    # Daftar rasio testing yang akan dicoba
    test_sizes = {
        '90_10': 0.10,
        '80_20': 0.20,
        '70_30': 0.30,
        '60_40': 0.40
    }
    
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            continue
            
        print(f"\n--- Skenario {key.upper()} ---")
        df = pd.read_csv(path)
        
        for split_name, test_size in test_sizes.items():
            train_ratio = int(100 - (test_size * 100))
            test_ratio = int(test_size * 100)
            
            len_train, len_test = split_and_save(df, TARGET, test_size, RANDOM_STATE, key, split_name)
            
            print(f"Rasio {train_ratio}:{test_ratio} -> Train: {len_train} | Test: {len_test}")

if __name__ == "__main__":
    main()
