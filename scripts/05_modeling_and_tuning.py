import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import xgboost as xgb
except ImportError:
    print("Pip install xgboost diperlukan!")
    import sys
    sys.exit(1)

def evaluate_model(y_true_asli, y_pred_asli):
    mae = mean_absolute_error(y_true_asli, y_pred_asli)
    rmse = np.sqrt(mean_squared_error(y_true_asli, y_pred_asli))
    r2 = r2_score(y_true_asli, y_pred_asli)
    return mae, rmse, r2

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', '04_preprocessed_data.csv')
    model_dir = os.path.join(script_dir, '..', 'models')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print("=== TAHAP 5: MODELING & HYPERPARAMETER TUNING ===")
    df = pd.read_csv(data_path)
    
    # 1. Pisahkan Fitur (X) dan Target (y)
    # y yang dipelajari adalah log_harga. Harga asli disimpan hanya untuk evaluasi tes.
    X = df.drop(columns=['harga', 'log_harga'])
    y_log = df['log_harga']
    y_asli = df['harga'] # Untuk evaluasi metrik (anti-log)
    
    # Simpan nama kolom fitur untuk tahap prediksi di akhir
    columns_path = os.path.join(model_dir, 'model_columns.json')
    with open(columns_path, 'w') as f:
        json.dump(list(X.columns), f)
        
    # 2. Pembagian Data (Splitting) 90:10
    X_train, X_test, y_train_log, y_test_log, _, y_test_asli = train_test_split(
        X, y_log, y_asli, test_size=0.1, random_state=42
    )
    print(f"Total Data Latih (Train): {len(X_train)}")
    print(f"Total Data Uji (Test)  : {len(X_test)}")
    
    # 3. Tuning Hyperparameter dengan GridSearch
    print("\n[Proses] Sedang mencari kombinasi parameter XGBoost terbaik...")
    param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [300, 500]
    }
    
    xgb_base = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    grid_search = GridSearchCV(
        estimator=xgb_base, 
        param_grid=param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1,
        verbose=1
    )
    
    # Train dengan target berwujud Logaritma
    grid_search.fit(X_train, y_train_log)
    best_model = grid_search.best_estimator_
    
    print("\nParameter Terbaik Ditemukan:")
    for k, v in grid_search.best_params_.items():
        print(f" - {k}: {v}")
        
    # 4. Evaluasi Model pada Data Test
    # Tebakan model berupa nilai logaritma
    y_pred_log = best_model.predict(X_test)
    
    # ANTI-LOG / INVERSE: Ubah nilai log menjadi nilai rupiah agar RMSE masuk akal
    y_pred_asli = np.expm1(y_pred_log)
    
    # Bandingkan Tebakan (Rupiah) dengan Jawaban Asli (Rupiah)
    mae, rmse, r2 = evaluate_model(y_test_asli, y_pred_asli)
    
    print("\n--- HASIL EVALUASI AKHIR (DATA UJI) ---")
    print(f"R-Squared : {r2:.4f} ({(r2*100):.2f}%)")
    print(f"MAE       : Rp {mae:,.0f}")
    print(f"RMSE      : Rp {rmse:,.0f}")
    
    # 5. Simpan Model Permanen
    model_path = os.path.join(model_dir, 'xgboost_kost_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n[SUKSES] Model Cerdas telah diexport ke: {model_path}")

if __name__ == "__main__":
    main()
