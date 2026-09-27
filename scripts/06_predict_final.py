import pandas as pd
import numpy as np
import os
import joblib
import json

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '..', 'models')
    
    model_path = os.path.join(model_dir, 'xgboost_kost_model.pkl')
    columns_path = os.path.join(model_dir, 'model_columns.json')
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        print("[ERROR] Model final belum dibuat! Jalankan script 05 terlebih dahulu.")
        return
        
    # 1. Load Model dan Struktur Kolom
    print("Memuat AI Model XGBoost dari file .pkl...")
    model = joblib.load(model_path)
    
    with open(columns_path, 'r') as f:
        train_columns = json.load(f)
        
    print("\n" + "="*45)
    print("   APLIKASI PREDIKSI HARGA KOST (XGBOOST)")
    print("="*45)
    print("Silakan masukkan data kost yang ingin ditebak:")
    
    try:
        # 2. Input Interaktif
        wilayah = input("1. Wilayah (contoh: Kota Surabaya)  : ")
        kecamatan = input("2. Kecamatan (contoh: Gubeng)       : ")
        jenis = input("3. Jenis Kost (Putra/Putri/Campur)  : ")
        rating = float(input("4. Rating Kost (0.0 s/d 5.0)        : "))
        ac = input("5. Ada AC? (Ya/Tidak)               : ")
        wifi = input("6. Ada WiFi? (Ya/Tidak)             : ")
        kasur = input("7. Ada Kasur? (Ya/Tidak)            : ")
        kamar_mandi = input("8. Kamar Mandi Dalam? (Ya/Tidak)    : ")
        kloset = input("9. Kloset Duduk? (Ya/Tidak)         : ")
        akses_24 = input("10. Akses 24 Jam? (Ya/Tidak)        : ")
        
        # 3. Proses Data Input agar sama persis dengan format waktu training
        input_data = pd.DataFrame(0, index=[0], columns=train_columns)
        
        # Mapping Ya/Tidak menjadi angka 1/0 untuk menghitung total_fasilitas
        ya_words = ['ya', 'ada', 'true', '1']
        
        is_ac = 1 if ac.strip().lower() in ya_words else 0
        is_wifi = 1 if wifi.strip().lower() in ya_words else 0
        is_kasur = 1 if kasur.strip().lower() in ya_words else 0
        is_km_dalam = 1 if kamar_mandi.strip().lower() in ya_words else 0
        is_kloset = 1 if kloset.strip().lower() in ya_words else 0
        is_akses24 = 1 if akses_24.strip().lower() in ya_words else 0
        
        input_data['rating'] = rating
        
        # Mapping Text ke kolom One-Hot Encoding
        def set_ohe(col_prefix, val):
            # Jika user ketik 'Ya', maka val='Ya'. Kita cari kolom 'ac_Ya'.
            val_clean = val.strip().title()
            # Khusus untuk ya/tidak, standarkan ke Ya/Tidak
            if val_clean.lower() in ya_words:
                val_clean = 'Ya'
            else:
                val_clean = 'Tidak'
                
            col_name = f"{col_prefix}_{val_clean}"
            if col_name in input_data.columns:
                input_data[col_name] = 1
                
        set_ohe('ac', ac)
        set_ohe('wifi', wifi)
        set_ohe('kasur', kasur)
        set_ohe('kamar_mandi_dalam', kamar_mandi)
        set_ohe('kloset_duduk', kloset)
        set_ohe('akses_24_jam', akses_24)
        
        # Fitur Engineering buatan kita: total_fasilitas
        input_data['total_fasilitas'] = is_ac + is_wifi + is_kasur + is_km_dalam + is_kloset + is_akses24
        
        # Mapping Text wilayah/kecamatan ke kolom One-Hot Encoding
        wilayah_col = f"wilayah_{wilayah.strip()}"
        if wilayah_col in input_data.columns:
            input_data[wilayah_col] = 1
            
        kecamatan_col = f"kecamatan_{kecamatan.strip()}"
        if kecamatan_col in input_data.columns:
            input_data[kecamatan_col] = 1
            
        jenis_col = f"jenis_{jenis.strip().title()}"
        if jenis_col in input_data.columns:
            input_data[jenis_col] = 1
            
        # 4. Melakukan Prediksi (Hasil mentah berupa nilai Logaritma)
        prediksi_log = model.predict(input_data)[0]
        
        # 5. Anti-Log (Exponensial) untuk membalikkan angka jadi Rupiah asli
        prediksi_rupiah = np.expm1(prediksi_log)
        
        print("\n" + "-"*45)
        print("MENGHITUNG PREDIKSI HARGA...")
        print("-" * 45)
        print(f"Harga Wajar Kost: ~ Rp {prediksi_rupiah:,.0f} per bulan")
        print("="*45)
        
    except ValueError as e:
        print("\n[ERROR] Terjadi kesalahan:")
        print(e)
    except Exception as e:
        print(f"\n[ERROR LAINNYA]: {e}")

if __name__ == '__main__':
    main()
