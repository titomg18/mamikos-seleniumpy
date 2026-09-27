import pandas as pd
import os

def select_features(df):
    """
    Melakukan seleksi fitur dengan menghapus kolom yang tidak relevan
    untuk pemodelan, seperti nama kost dan url (identifier unik).
    """
    # Kolom yang diidentifikasi tidak relevan sebagai prediktor harga
    cols_to_drop = ['nama', 'url']
    
    # Menghapus kolom jika ada di dataset
    df_selected = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # === FEATURE ENGINEERING ===
    # Menambahkan fitur 'total_fasilitas' yang merepresentasikan jumlah fasilitas yang ada
    fasilitas_cols = ['ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
    # Kita hitung berapa banyak kata 'Ya' / 'Ada' pada kolom fasilitas tersebut
    df_selected['total_fasilitas'] = 0
    for f in fasilitas_cols:
        if f in df_selected.columns:
            # Karena isinya string ('Ya', 'Tidak'), kita cek stringnya
            df_selected['total_fasilitas'] += df_selected[f].astype(str).str.strip().str.lower().isin(['ya', 'ada']).astype(int)
            
    
    return df_selected, cols_to_drop

def main():
    # Path file dari tahap sebelumnya
    path_drop = r'..\data\processed\01_cleaned_drop.csv'
    path_impute = r'..\data\processed\01_cleaned_imputed.csv'
    
    # Cek ketersediaan file
    if not os.path.exists(path_drop) or not os.path.exists(path_impute):
        print("Dataset dari tahap 1 tidak ditemukan.")
        return
        
    print("=== PROSES SELEKSI FITUR ===")
    
    # 1. Proses Skenario DROP
    df_drop = pd.read_csv(path_drop)
    print(f"\nSkenario DROP - Kolom awal ({len(df_drop.columns)}): {list(df_drop.columns)}")
    
    df_drop_selected, dropped_cols = select_features(df_drop)
    out_path_drop = r'..\data\processed\02_selected_drop.csv'
    df_drop_selected.to_csv(out_path_drop, index=False)
    
    print(f"Kolom yang dihapus: {dropped_cols}")
    print(f"Kolom terpilih ({len(df_drop_selected.columns)}): {list(df_drop_selected.columns)}")
    print(f"-> Skenario DROP disimpan di: {out_path_drop}")
    
    # 2. Proses Skenario IMPUTE
    df_impute = pd.read_csv(path_impute)
    print(f"\nSkenario IMPUTE - Kolom awal ({len(df_impute.columns)}): {list(df_impute.columns)}")
    
    df_impute_selected, dropped_cols = select_features(df_impute)
    out_path_impute = r'..\data\processed\02_selected_imputed.csv'
    df_impute_selected.to_csv(out_path_impute, index=False)
    
    print(f"Kolom yang dihapus: {dropped_cols}")
    print(f"Kolom terpilih ({len(df_impute_selected.columns)}): {list(df_impute_selected.columns)}")
    print(f"-> Skenario IMPUTE disimpan di: {out_path_impute}")

if __name__ == "__main__":
    main()
