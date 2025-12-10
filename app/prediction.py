import os
import joblib
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_dir, 'modelling/saved_models')

model = joblib.load(os.path.join(models_path, 'model_rf.pkl'))
encoder = joblib.load(os.path.join(models_path, 'encoder_kabupaten.pkl'))
feature_cols = joblib.load(os.path.join(models_path, 'feature_columns.pkl'))

def predict_hujan(tanggal_str, kabupaten):
    try:
        t = datetime.strptime(tanggal_str, "%Y-%m-%d")
    except:
        t = datetime.strptime(tanggal_str, "%d-%m-%Y")

    bulan = t.month
    tanggal_hari = t.day

    kab_encoded = encoder.transform(pd.DataFrame({"Kabupaten": [kabupaten]}))
    kab_cols = encoder.get_feature_names_out(["Kabupaten"])
    df_kab = pd.DataFrame(kab_encoded.toarray(), columns=kab_cols)

    fitur_lain = pd.DataFrame([{
        "Bulan": bulan,
        "Tanggal": tanggal_hari
    }])

    X_new = pd.concat([fitur_lain, df_kab], axis=1)
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)

    pred = model.predict(X_new)[0]
    return "Hujan" if pred == 1 else "Tidak Hujan"

if __name__ == "__main__":
    print("=== Prediksi Hujan Berdasarkan Tanggal & Kabupaten ===")
    tanggal = input("Masukkan tanggal (YYYY-MM-DD): ")
    kab = input("Masukkan nama kabupaten: ")

    hasil = predict_hujan(tanggal, kab)
    
    print("-" * 50)
    print(f"Prediksi {tanggal} di {kab}: {hasil}")
    print("-" * 50)