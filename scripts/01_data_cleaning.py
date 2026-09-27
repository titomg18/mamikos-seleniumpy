import pandas as pd
import numpy as np
import os

def clean_data(df):
    df_clean = df.copy()
    
    print(f"Data awal: {len(df_clean)} baris")
    
    # 1. Hapus Duplikasi (Penting untuk data web scraping)
    df_clean.drop_duplicates(inplace=True)
    print(f"Setelah hapus duplikat: {len(df_clean)} baris")
    
    # 2. Standarisasi Tipe Data Numerik
    for col in ['harga', 'rating']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    # 3. Hapus baris yang Target Prediksinya (Harga) Kosong atau Negatif/0
    if 'harga' in df_clean.columns:
        df_clean.dropna(subset=['harga'], inplace=True)
        df_clean = df_clean[df_clean['harga'] > 0]
        
    print(f"Setelah hapus harga kosong/invalid: {len(df_clean)} baris")
    
    # 4. Tangani Nilai Kosong (Missing Values) pada Fitur Lain
    # Strategi terbaik berdasarkan evaluasi sebelumnya: DROP baris yang bolong
    df_clean.dropna(inplace=True)
    print(f"Setelah hapus missing values lainnya: {len(df_clean)} baris")
    
    # 5. Standarisasi Teks Kategorikal agar seragam
    if 'jenis' in df_clean.columns:
        df_clean['jenis'] = df_clean['jenis'].astype(str).str.strip().str.title()
        
    # 6. Menangani Outlier Harga Ekstrem (Metode IQR)
    # Ini penting agar kost super mewah atau kos abal-abal tidak membingungkan AI
    Q1 = df_clean['harga'].quantile(0.25)
    Q3 = df_clean['harga'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_condition = (df_clean['harga'] < lower_bound) | (df_clean['harga'] > upper_bound)
    df_clean = df_clean[~outlier_condition]
    print(f"Setelah hapus outlier harga (Sisa Data Bersih): {len(df_clean)} baris")
    
    return df_clean

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, '..', 'data', 'raw', 'dataset.csv')
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    # Buat folder processed jika belum ada
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    if not os.path.exists(raw_path):
        print(f"[ERROR] File mentah tidak ditemukan di {raw_path}")
        return
        
    print("=== TAHAP 1: PEMBERSIHAN DATA (DATA CLEANING) ===")
    df = pd.read_csv(raw_path)
    
    df_clean = clean_data(df)
    
    # Simpan hasil
    out_path = os.path.join(processed_dir, '01_cleaned_data.csv')
    df_clean.to_csv(out_path, index=False)
    print(f"\n[SUKSES] Data bersih berhasil disimpan di: {out_path}")

if __name__ == "__main__":
    main()
