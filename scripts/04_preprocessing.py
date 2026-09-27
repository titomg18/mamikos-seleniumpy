import pandas as pd
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', '02_engineered_data.csv')
    out_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data tidak ditemukan di {data_path}")
        return
        
    print("=== TAHAP 4: PREPROCESSING (ONE-HOT ENCODING) ===")
    df = pd.read_csv(data_path)
    
    # Pisahkan kolom target (kita simpan log_harga dan harga aslinya sementara)
    # Harga asli disimpan sekadar referensi bila dibutuhkan, tapi tidak di-train
    
    # Kolom kategorikal yang akan di-encode
    cat_cols = ['wilayah', 'kecamatan', 'jenis', 'ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
    
    # Lakukan One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns])
    
    # Ubah True/False menjadi 1/0 agar XGBoost lebih mudah memproses
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    # Pastikan 'harga' asli tidak dimasukkan sebagai fitur ML
    # Model HANYA boleh memprediksi 'log_harga'
    
    out_path = os.path.join(out_dir, '04_preprocessed_data.csv')
    df_encoded.to_csv(out_path, index=False)
    
    print(f"Data awal memiliki {len(df.columns)} kolom.")
    print(f"Setelah One-Hot Encoding, data memiliki {len(df_encoded.columns)} kolom.")
    print(f"[SUKSES] Data siap-ML disimpan di: {out_path}")

if __name__ == "__main__":
    main()
