import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost belum terinstal. Silakan jalankan: pip install xgboost")
    import sys
    sys.exit(1)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    scenarios = ['drop', 'impute']
    test_sizes = ['90_10', '80_20', '70_30', '60_40']
    
    results = []
    
    print("=== PROSES PELATIHAN MODEL XGBOOST ===")
    
    for scenario in scenarios:
        for size in test_sizes:
            # Menyusun path file
            X_train_path = os.path.join(data_dir, f'X_train_{scenario}_{size}.csv')
            X_test_path = os.path.join(data_dir, f'X_test_{scenario}_{size}.csv')
            y_train_path = os.path.join(data_dir, f'y_train_{scenario}_{size}.csv')
            y_test_path = os.path.join(data_dir, f'y_test_{scenario}_{size}.csv')
            
            # Cek jika ada file yang hilang
            if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
                print(f"Data tidak lengkap untuk skenario {scenario} rasio {size}. Dilewati.")
                continue
                
            # Load Data
            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path)
            y_train = pd.read_csv(y_train_path).squeeze() # Pastikan bentuknya 1D Series
            y_test = pd.read_csv(y_test_path).squeeze()
            
            # Transformasi Target (Harga) dengan Logaritma (log1p)
            y_train_log = np.log1p(y_train)
            
            # Inisialisasi Model XGBoost
            model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
            
            # Latih Model menggunakan target Logaritma
            model.fit(X_train, y_train_log)
            
            # Lakukan Prediksi pada data Test dan kembalikan ke skala harga asli (expm1)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
            # Evaluasi Akurasi Prediksi
            mae, rmse, r2 = evaluate_model(y_test, y_pred)
            
            # Simpan hasil
            results.append({
                'Skenario': scenario.upper(),
                'Rasio': size.replace('_', ':'),
                'MAE (Rp)': round(mae, 2),
                'RMSE (Rp)': round(rmse, 2),
                'R-Squared': round(r2, 4)
            })
            print(f"Selesai dilatih: {scenario.upper()} | Rasio {size.replace('_', ':')} -> R2: {r2:.4f}")
            
    # Buat DataFrame dari hasil
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("Tidak ada model yang berhasil dilatih.")
        return
        
    # Urutkan berdasarkan nilai R-Squared (mendekati 1 semakin bagus)
    df_results = df_results.sort_values(by='R-Squared', ascending=False)
    
    # Simpan hasil ke CSV
    results_path = os.path.join(data_dir, '05_evaluation_results.csv')
    df_results.to_csv(results_path, index=False)
    
    print("\n--- RINGKASAN HASIL EVALUASI (Diurutkan dari Terbaik) ---")
    print(df_results.to_string(index=False))
    print(f"\nTabel evaluasi lengkap disimpan di: {results_path}")

if __name__ == '__main__':
    main()
