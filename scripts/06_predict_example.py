import pandas as pd
import os
import xgboost as xgb

def create_prediction_input(user_input, train_columns):
    """
    Fungsi untuk mengubah input manusia ke dalam format angka (One-Hot Encoding)
    agar bisa dibaca oleh model XGBoost.
    """
    # Buat dataframe dengan 1 baris, semua kolom diisi 0
    input_data = pd.DataFrame(0, index=[0], columns=train_columns)
    
    # 1. Isi fitur numerik & biner
    input_data['rating'] = user_input.get('rating', 0)
    input_data['ac'] = 1 if str(user_input.get('ac')).lower() == 'ya' else 0
    input_data['akses_24_jam'] = 1 if str(user_input.get('akses_24_jam')).lower() == 'ya' else 0
    input_data['kasur'] = 1 if str(user_input.get('kasur')).lower() == 'ya' else 0
    input_data['kloset_duduk'] = 1 if str(user_input.get('kloset_duduk')).lower() == 'ya' else 0
    input_data['kamar_mandi_dalam'] = 1 if str(user_input.get('kamar_mandi_dalam')).lower() == 'ya' else 0
    input_data['wifi'] = 1 if str(user_input.get('wifi')).lower() == 'ya' else 0
    
    # 2. Isi fitur kategorikal (One-Hot Encoding)
    # Mengaktifkan kolom spesifik menjadi 1
    wilayah_col = f"wilayah_{user_input.get('wilayah')}"
    if wilayah_col in input_data.columns:
        input_data[wilayah_col] = 1
        
    kecamatan_col = f"kecamatan_{user_input.get('kecamatan')}"
    if kecamatan_col in input_data.columns:
        input_data[kecamatan_col] = 1
        
    jenis_col = f"jenis_{user_input.get('jenis')}"
    if jenis_col in input_data.columns:
        input_data[jenis_col] = 1
        
    return input_data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    
    # Kita gunakan model terbaik dari eksperimen: Skenario DROP rasio 90:10
    X_train_path = os.path.join(data_dir, 'X_train_drop_90_10.csv')
    y_train_path = os.path.join(data_dir, 'y_train_drop_90_10.csv')
    
    if not os.path.exists(X_train_path):
        print("Data train tidak ditemukan. Pastikan proses sebelumnya sudah berhasil.")
        return

    # 1. Load Data Terbaik dan Latih Ulang Model
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    
    model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    print("=== SIMULASI PREDIKSI HARGA KOST ===")
    print("Model XGBoost siap digunakan!\n")
    
    # 2. Input Kos Sesuai Contoh Peneliti
    # (Catatan: "Kamar mandi : Dalam" diubah menjadi kamar_mandi_dalam : 'Ya')
    contoh_input = {
        'wilayah': 'Kota Surabaya',
        'jenis': 'Putra',
        'rating': 5.0,
        'ac': 'Tidak',
        'wifi': 'Ya',
        'kasur': 'Ya',
        'kamar_mandi_dalam': 'Tidak',
        'akses_24_jam': 'Ya',
        'kloset_duduk': 'Ya', # Asumsi fasilitas standar ada
        'kecamatan': 'Gubeng' # Kecamatan random untuk kelengkapan data
    }
    
    print("--- Data Karakteristik Kost ---")
    for k, v in contoh_input.items():
        print(f"{k.ljust(20)}: {v}")
        
    # 3. Proses Prediksi
    X_pred = create_prediction_input(contoh_input, X_train.columns)
    prediksi = model.predict(X_pred)[0]
    
    print("\n-------------------------------")
    print(f"OUTPUT PREDIKSI HARGA OLEH XGBOOST:")
    print(f"~ Rp {prediksi:,.0f} / bulan")
    print("-------------------------------")

if __name__ == '__main__':
    main()
