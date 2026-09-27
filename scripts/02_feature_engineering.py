import pandas as pd
import numpy as np
import os

def feature_engineering(df):
    df_engineered = df.copy()
    
    # 1. Hapus Fitur yang Tidak Berpengaruh ke Harga (Identifier)
    cols_to_drop = ['nama', 'url']
    df_engineered.drop(columns=[col for col in cols_to_drop if col in df_engineered.columns], inplace=True)
    
    # 2. Rekayasa Fitur (Feature Engineering): Total Fasilitas
    # Semakin banyak fasilitas, biasanya harga semakin mahal. Kita buat kolom rangkumannya.
    fasilitas_cols = ['ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
    df_engineered['total_fasilitas'] = 0
    for f in fasilitas_cols:
        if f in df_engineered.columns:
            # Hitung jumlah jawaban 'Ya' atau 'Ada'
            df_engineered['total_fasilitas'] += df_engineered[f].astype(str).str.strip().str.lower().isin(['ya', 'ada']).astype(int)
            
    # 3. TRANSFORMASI LOGARITMIK PADA TARGET (KUNCI AKURASI)
    # Kita buat kolom 'log_harga' untuk dipelajari model, agar kurva distribusi harga lebih normal (tidak menceng)
    if 'harga' in df_engineered.columns:
        df_engineered['log_harga'] = np.log1p(df_engineered['harga'])
        
    return df_engineered

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(script_dir, '..', 'data', 'processed', '01_cleaned_data.csv')
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    if not os.path.exists(clean_path):
        print(f"[ERROR] Data bersih tidak ditemukan di {clean_path}")
        return
        
    print("=== TAHAP 2: FEATURE ENGINEERING & LOGARITMA ===")
    df = pd.read_csv(clean_path)
    
    df_eng = feature_engineering(df)
    
    # Simpan hasil
    out_path = os.path.join(processed_dir, '02_engineered_data.csv')
    df_eng.to_csv(out_path, index=False)
    
    print("Berhasil menambahkan kolom 'total_fasilitas'.")
    print("Berhasil membuat target 'log_harga'.")
    print(f"[SUKSES] Data disimpan di: {out_path}")

if __name__ == "__main__":
    main()
