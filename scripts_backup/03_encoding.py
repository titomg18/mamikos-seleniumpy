import pandas as pd
import os

def encode_data(df):
    df_encoded = df.copy()
    
    # 1. Encoding biner untuk kolom (Ya / Tidak)
    binary_cols = ['ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
    
    for col in binary_cols:
        if col in df_encoded.columns:
            # Ubah ke huruf kecil dan bersihkan spasi untuk memastikan keseragaman
            df_encoded[col] = df_encoded[col].astype(str).str.strip().str.lower()
            df_encoded[col] = df_encoded[col].map({'ya': 1, 'tidak': 0})
            
            # Isi NaN dengan 0
            df_encoded[col] = df_encoded[col].fillna(0).astype(int)
            
    # 2. One-Hot Encoding (OHE) untuk kolom kategorikal
    # Menghasilkan 1 dan 0 pada kolom-kolom baru (e.g., jenis_Putra, jenis_Putri, dsb)
    categorical_cols = ['wilayah', 'kecamatan', 'jenis']
    existing_cat_cols = [col for col in categorical_cols if col in df_encoded.columns]
    
    # pd.get_dummies dengan dtype=int agar outputnya 0 dan 1 (bukan True/False)
    df_encoded = pd.get_dummies(df_encoded, columns=existing_cat_cols, dtype=int)
            
    return df_encoded

def main():
    paths = {
        'drop': r'..\data\processed\02_selected_drop.csv',
        'impute': r'..\data\processed\02_selected_imputed.csv'
    }
    
    print("=== PROSES ENCODING (ONE-HOT ENCODING) ===")
    
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            continue
            
        print(f"\nMemproses Skenario {key.upper()}...")
        df = pd.read_csv(path)
        
        df_encoded = encode_data(df)
        
        out_path = rf'..\data\processed\03_encoded_{key}.csv'
        df_encoded.to_csv(out_path, index=False)
        
        print(f"-> Total Kolom setelah OHE: {len(df_encoded.columns)}")
        print(f"-> Sampel Kolom: {list(df_encoded.columns)[:10]} ...")
        print(f"-> Disimpan di: {out_path}")

if __name__ == "__main__":
    main()
