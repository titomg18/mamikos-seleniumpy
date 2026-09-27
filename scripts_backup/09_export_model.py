import pandas as pd
import xgboost as xgb
import joblib
import json
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    model_dir = os.path.join(script_dir, '..', 'models')
    
    # Buat folder 'models' jika belum ada
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print("=== PROSES EXPORT MODEL XGBOOST ===")
    
    # 1. Load Data Pelatihan Terbaik (Skenario Drop 90:10)
    X_train_path = os.path.join(data_dir, 'X_train_drop_90_10.csv')
    y_train_path = os.path.join(data_dir, 'y_train_drop_90_10.csv')
    
    if not os.path.exists(X_train_path):
        print("File data latih tidak ditemukan!")
        return
        
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    
    print("Melatih model dengan Parameter Terbaik (Hasil Tuning)...")
    
    # 2. Inisiasi Model dengan Parameter Terbaik
    best_model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=7,
        n_estimators=500,
        subsample=1.0,
        random_state=42,
        objective='reg:squarederror'
    )
    
    import numpy as np
    y_train_log = np.log1p(y_train)
    
    # Latih model secara final menggunakan target log
    best_model.fit(X_train, y_train_log)
    
    # 3. Simpan Model Permanen
    model_path = os.path.join(model_dir, 'xgboost_kost_model.pkl')
    joblib.dump(best_model, model_path)
    
    # 4. Simpan Struktur Kolom (Sangat Penting untuk Deploy)
    # Agar saat nanti membuat prediksi baru, urutan fiturnya tidak tertukar
    columns_path = os.path.join(model_dir, 'model_columns.json')
    with open(columns_path, 'w') as f:
        json.dump(list(X_train.columns), f)
        
    print(f"\n[OK] BERHASIL!")
    print(f"Model permanen tersimpan di : {model_path}")
    print(f"Struktur kolom tersimpan di : {columns_path}")
    print("Model ini sudah siap dipanggil kapan saja tanpa perlu dilatih ulang!")

if __name__ == '__main__':
    main()
