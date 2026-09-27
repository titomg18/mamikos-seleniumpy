import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', '02_engineered_data.csv')
    out_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data {data_path} tidak ditemukan.")
        return
        
    print("=== TAHAP 3: EXPLORATORY DATA ANALYSIS (EDA) ===")
    df = pd.read_csv(data_path)
    
    # 1. Distribusi Harga Asli vs Log Harga
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(df['harga'], bins=50, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribusi Harga Kost Asli (Menceng)')
    axes[0].set_xlabel('Harga (Rp)')
    axes[0].set_ylabel('Frekuensi')
    
    sns.histplot(df['log_harga'], bins=50, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Distribusi Log(Harga Kost) (Lebih Normal)')
    axes[1].set_xlabel('Log(Harga)')
    axes[1].set_ylabel('Frekuensi')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eda_1_distribusi_harga.png'), dpi=300)
    plt.close()
    print("- Grafik Distribusi Harga berhasil dibuat.")
    
    # 2. Boxplot Harga vs Jenis Kost
    if 'jenis' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='jenis', y='harga', data=df)
        plt.title('Sebaran Harga Berdasarkan Jenis Kost')
        plt.xlabel('Jenis Kost')
        plt.ylabel('Harga (Rp)')
        plt.savefig(os.path.join(out_dir, 'eda_2_harga_vs_jenis.png'), dpi=300)
        plt.close()
        print("- Grafik Harga vs Jenis Kost berhasil dibuat.")
        
    # 3. Heatmap Korelasi
    df_corr = df.copy()
    fasilitas = ['ac', 'akses_24_jam', 'kasur', 'kloset_duduk', 'kamar_mandi_dalam', 'wifi']
    for f in fasilitas:
        if f in df_corr.columns:
            df_corr[f] = df_corr[f].map({'Ya': 1, 'Tidak': 0, 'Ada': 1, 'Tidak Ada': 0})
            
    num_cols = df_corr.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriks Korelasi Harga dan Fasilitas')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eda_3_heatmap_korelasi.png'), dpi=300)
    plt.close()
    print("- Grafik Heatmap Korelasi berhasil dibuat.")
    
    print("\n[SUKSES] Semua grafik EDA tersimpan di folder data/processed/")

if __name__ == '__main__':
    main()
