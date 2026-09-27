import pandas as pd
import numpy as np
import os

def process_data(df, handle_missing='drop', handle_outlier='drop'):
    df_clean = df.copy()
    
    # 1. Hapus duplikasi (selalu dilakukan untuk data identik)
    df_clean.drop_duplicates(inplace=True)
    
    # Pastikan tipe data harga dan rating numerik
    if 'harga' in df_clean.columns:
        df_clean['harga'] = pd.to_numeric(df_clean['harga'], errors='coerce')
    if 'rating' in df_clean.columns:
        df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
        
    # 2. Tangani Nilai Kosong (Missing Values)
    # Jika harga kosong, selalu kita hapus karena itu target variabel
    if 'harga' in df_clean.columns:
        df_clean.dropna(subset=['harga'], inplace=True)
    
    missing_cols = df_clean.columns[df_clean.isnull().any()]
    
    if handle_missing == 'drop':
        df_clean.dropna(inplace=True)
    elif handle_missing == 'impute':
        for col in missing_cols:
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                # Imputasi dengan Modus untuk kategorikal
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
            else:
                # Imputasi dengan Median untuk numerikal
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                
    # 3. Data Tidak Valid
    # Harga <= 0 (selalu dihapus karena harga tidak mungkin gratis/negatif)
    df_clean = df_clean[df_clean['harga'] > 0]
    
    # Rating harus antara 0 - 5
    if 'rating' in df_clean.columns:
        df_clean = df_clean[(df_clean['rating'] >= 0) & (df_clean['rating'] <= 5) | df_clean['rating'].isna()]
        
    # Konsistensi kategori jenis
    if 'jenis' in df_clean.columns:
        df_clean['jenis'] = df_clean['jenis'].astype(str).str.strip().str.title()
        
    # 4. Tangani Outlier Harga
    Q1 = df_clean['harga'].quantile(0.25)
    Q3 = df_clean['harga'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_condition = (df_clean['harga'] < lower_bound) | (df_clean['harga'] > upper_bound)
    
    if handle_outlier == 'drop':
        # Hapus outlier
        df_clean = df_clean[~outlier_condition]
    elif handle_outlier == 'impute':
        # Ganti dengan median
        median_harga = df_clean['harga'].median()
        df_clean.loc[outlier_condition, 'harga'] = median_harga
        
    return df_clean

def main():
    raw_path = r'..\data\raw\dataset.csv'
    
    if not os.path.exists(raw_path):
        print(f"File tidak ditemukan di {raw_path}")
        return
        
    df = pd.read_csv(raw_path)
    print(f"Jumlah data awal: {len(df)} baris")
    
    # Kondisi 1: Drop Missing Value & Drop Outlier
    df_drop = process_data(df, handle_missing='drop', handle_outlier='drop')
    path_drop = r'..\data\processed\01_cleaned_drop.csv'
    df_drop.to_csv(path_drop, index=False)
    print(f"-> Skenario DROP: Disimpan ke {path_drop} ({len(df_drop)} baris)")
    
    # Kondisi 2: Impute Missing Value (Modus/Median) & Impute Outlier (Median)
    df_impute = process_data(df, handle_missing='impute', handle_outlier='impute')
    path_impute = r'..\data\processed\01_cleaned_imputed.csv'
    df_impute.to_csv(path_impute, index=False)
    print(f"-> Skenario IMPUTE: Disimpan ke {path_impute} ({len(df_impute)} baris)")

if __name__ == "__main__":
    main()
