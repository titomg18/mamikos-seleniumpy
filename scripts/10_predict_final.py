import pandas as pd
import os
import joblib
import json

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '..', 'models')
    
    model_path = os.path.join(model_dir, 'xgboost_kost_model.pkl')
    columns_path = os.path.join(model_dir, 'model_columns.json')
    
    # 1. Pastikan model final sudah ada
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        print("Model final belum dibuat! Jalankan 09_export_model.py terlebih dahulu.")
        return
        
    # 2. Load Model dan Struktur Kolom
    print("Memuat AI Model dari file .pkl...")
    model = joblib.load(model_path)
    
    with open(columns_path, 'r') as f:
        train_columns = json.load(f)
        
    print("\n" + "="*45)
    print("   APLIKASI PREDIKSI HARGA KOST (XGBOOST)")
    print("="*45)
    print("Silakan masukkan data kost yang ingin ditebak:")
    
    # 3. Input Interaktif dari Pengguna (Terminal)
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
    
    # 4. Memproses Input menjadi Format AI (One-Hot Encoding dll)
    input_data = pd.DataFrame(0, index=[0], columns=train_columns)
    
    input_data['rating'] = rating
    input_data['ac'] = 1 if ac.strip().lower() == 'ya' else 0
    input_data['wifi'] = 1 if wifi.strip().lower() == 'ya' else 0
    input_data['kasur'] = 1 if kasur.strip().lower() == 'ya' else 0
    input_data['kamar_mandi_dalam'] = 1 if kamar_mandi.strip().lower() == 'ya' else 0
    input_data['kloset_duduk'] = 1 if kloset.strip().lower() == 'ya' else 0
    input_data['akses_24_jam'] = 1 if akses_24.strip().lower() == 'ya' else 0
    
    wilayah_col = f"wilayah_{wilayah.strip()}"
    if wilayah_col in input_data.columns:
        input_data[wilayah_col] = 1
        
    kecamatan_col = f"kecamatan_{kecamatan.strip()}"
    if kecamatan_col in input_data.columns:
        input_data[kecamatan_col] = 1
        
    jenis_col = f"jenis_{jenis.strip()}"
    if jenis_col in input_data.columns:
        input_data[jenis_col] = 1
        
    # 5. Prediksi Harga
    prediksi = model.predict(input_data)[0]
    
    print("\n" + "-"*45)
    print("MENGHITUNG PREDIKSI HARGA...")
    print("-"*45)
    print(f"Harga Wajar Kost: ~ Rp {prediksi:,.0f} per bulan")
    print("="*45)

if __name__ == '__main__':
    # Pastikan error terkait nilai tidak dikenali ditangkap
    try:
        main()
    except ValueError:
        print("\n[ERROR] Pastikan Anda memasukkan angka pada isian Rating Kost (contoh: 4.5).")
