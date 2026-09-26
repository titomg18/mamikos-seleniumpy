import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    # Load dataset terbaik kita (Drop 90:10)
    X_train_path = os.path.join(data_dir, 'X_train_drop_90_10.csv')
    y_train_path = os.path.join(data_dir, 'y_train_drop_90_10.csv')
    X_test_path = os.path.join(data_dir, 'X_test_drop_90_10.csv')
    y_test_path = os.path.join(data_dir, 'y_test_drop_90_10.csv')
    
    if not os.path.exists(X_train_path):
        print("Data latih tidak ditemukan. Pastikan proses sebelumnya berhasil.")
        return
        
    print("=== PROSES HYPERPARAMETER TUNING XGBOOST ===")
    print("Mohon tunggu, model sedang mencoba berbagai kombinasi settingan...\n")
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # 1. Cek Skor Model Default (Tanpa Tuning)
    model_default = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    model_default.fit(X_train, y_train)
    y_pred_default = model_default.predict(X_test)
    r2_default = r2_score(y_test, y_pred_default)
    rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))
    
    # 2. Definisikan Kombinasi Parameter yang mau diuji
    param_grid = {
        'max_depth': [3, 5, 7],             # Kedalaman pohon keputusan
        'learning_rate': [0.01, 0.05, 0.1], # Seberapa cepat model belajar
        'n_estimators': [100, 300, 500],    # Jumlah pohon yang dibuat
        'subsample': [0.8, 1.0]             # Persentase sampel yang dipakai per pohon
    }
    
    # Inisiasi XGBoost
    xgb_estimator = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    
    # GridSearch akan mencoba semua (3x3x3x2 = 54 kombinasi) dikali 3 Fold Cross-Validation
    grid_search = GridSearchCV(
        estimator=xgb_estimator, 
        param_grid=param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1, # Gunakan semua CPU agar lebih cepat
        verbose=1
    )
    
    # Eksekusi pencarian
    grid_search.fit(X_train, y_train)
    
    # 3. Ambil Model Terbaik
    best_model = grid_search.best_estimator_
    
    # Prediksi menggunakan model terbaik
    y_pred_tuned = best_model.predict(X_test)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    
    print("\n--- HASIL TUNING (SEBELUM vs SESUDAH) ---")
    print(f"R-Squared Awal   : {r2_default:.4f}")
    print(f"R-Squared Baru   : {r2_tuned:.4f}  (Makin dekat 1 makin baik)")
    print(f"RMSE Awal        : Rp {rmse_default:,.0f}")
    print(f"RMSE Baru        : Rp {rmse_tuned:,.0f}  (Makin kecil makin baik)")
    
    print("\n--- SETTINGAN PARAMETER TERBAIK ---")
    best_params = grid_search.best_params_
    for k, v in best_params.items():
        print(f"{k.ljust(15)}: {v}")
        
    print("\nCatatan:")
    if r2_tuned > r2_default:
        print("Bagus! Tuning berhasil MENINGKATKAN akurasi model Anda.")
    else:
        print("Tuning selesai. Performa model sudah sangat optimal dari awal (tidak beda jauh).")

if __name__ == '__main__':
    main()
