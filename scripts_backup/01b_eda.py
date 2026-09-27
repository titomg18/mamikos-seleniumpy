import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', '01_cleaned_drop.csv')
    out_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    if not os.path.exists(data_path):
        print("Data belum dibersihkan. Jalankan 01_data_cleaning.py dulu.")
        return
        
    df = pd.read_csv(data_path)
    print("Memulai proses Exploratory Data Analysis (EDA)...")
    
    # 1. Distribusi Harga (Harga asli dan Log-Harga)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(df['harga'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Distribusi Harga Kost Asli')
    axes[0].set_xlabel('Harga (Rp)')
    axes[0].set_ylabel('Frekuensi')
    
    import numpy as np
    sns.histplot(np.log1p(df['harga']), bins=50, kde=True, ax=axes[1])
    axes[1].set_title('Distribusi Log(Harga Kost)')
    axes[1].set_xlabel('Log(Harga)')
    axes[1].set_ylabel('Frekuensi')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eda_1_distribusi_harga.png'))
    plt.close()
    
    # 2. Rata-rata Harga Berdasarkan Jenis Kost (Putra/Putri/Campur)
    if 'jenis' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='jenis', y='harga', data=df)
        plt.title('Sebaran Harga Berdasarkan Jenis Kost')
        plt.xlabel('Jenis Kost')
        plt.ylabel('Harga (Rp)')
        plt.savefig(os.path.join(out_dir, 'eda_2_harga_vs_jenis.png'))
        plt.close()
        
    # 3. Pengaruh AC terhadap Harga
    if 'ac' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='ac', y='harga', data=df)
        plt.title('Pengaruh Fasilitas AC Terhadap Harga Kost')
        plt.xlabel('Fasilitas AC')
        plt.ylabel('Harga (Rp)')
        plt.savefig(os.path.join(out_dir, 'eda_3_harga_vs_ac.png'))
        plt.close()

    # 4. Korelasi Fasilitas dengan Harga
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
    plt.savefig(os.path.join(out_dir, 'eda_4_heatmap_korelasi.png'))
    plt.close()
    
    print("Selesai! Grafik EDA telah disimpan di folder data/processed/")

if __name__ == '__main__':
    main()
