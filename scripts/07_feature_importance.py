import pandas as pd
import os
import xgboost as xgb
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Library matplotlib/seaborn belum terinstal. Silakan jalankan: pip install matplotlib seaborn")
    import sys
    sys.exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    # Kita gunakan data dari model terbaik (DROP 90:10)
    X_train_path = os.path.join(data_dir, 'X_train_drop_90_10.csv')
    y_train_path = os.path.join(data_dir, 'y_train_drop_90_10.csv')
    
    if not os.path.exists(X_train_path):
        print("Data train tidak ditemukan!")
        return

    # Load Data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    
    # Latih Model
    model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # AMBIL FEATURE IMPORTANCE
    # ---------------------------------------------------------
    # Dapatkan skor seberapa penting tiap kolom (fitur)
    importances = model.feature_importances_
    
    # Buat DataFrame agar mudah diurutkan
    df_importance = pd.DataFrame({
        'Fitur': X_train.columns,
        'Tingkat_Kepentingan': importances
    })
    
    # Urutkan dari yang paling penting
    df_importance = df_importance.sort_values(by='Tingkat_Kepentingan', ascending=False)
    
    # Karena fitur sangat banyak setelah OHE (>260), kita ambil Top 15 saja
    top_15 = df_importance.head(15)
    
    print("=== TOP 15 FAKTOR YANG MEMPENGARUHI HARGA KOST ===")
    for index, row in top_15.iterrows():
        print(f"{row['Fitur'].ljust(35)}: {row['Tingkat_Kepentingan']:.4f}")
        
    # ---------------------------------------------------------
    # VISUALISASI KE GRAFIK (Untuk ditaruh di Bab 4 Skripsi)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Tingkat_Kepentingan', y='Fitur', data=top_15, palette='viridis', hue='Fitur', legend=False)
    plt.title('Top 15 Faktor Penentu Harga Kost (XGBoost Feature Importance)', fontsize=14, fontweight='bold')
    plt.xlabel('Tingkat Kepentingan (Makin besar makin berpengaruh)', fontsize=12)
    plt.ylabel('Fitur/Karakteristik', fontsize=12)
    plt.tight_layout()
    
    # Simpan Gambar
    plot_path = os.path.join(data_dir, '07_feature_importance.png')
    plt.savefig(plot_path, dpi=300) # dpi=300 agar gambarnya HD/Tidak Pecah
    print(f"\n[OK] Grafik Feature Importance telah disimpan di: {plot_path}")
    print("Silakan buka gambar tersebut untuk dimasukkan ke laporan Skripsi Anda!")

if __name__ == '__main__':
    main()
