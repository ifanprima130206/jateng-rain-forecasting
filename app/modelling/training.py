import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
processed_path = os.path.join(project_root, 'dataset', 'processed')
models_path = os.path.join(current_dir, 'saved_models')

os.makedirs(models_path, exist_ok=True)

def train_model():
    data_file = os.path.join(processed_path, 'data_training_gabungan.csv')
    if not os.path.exists(data_file):
        print("Error: File data tidak ditemukan.")
        return

    df = pd.read_csv(data_file)
    df = df.dropna(subset=['Curah_Hujan', 'Kabupaten'])

    df["Tanggal_Full"] = pd.to_datetime(df["Tahun"].astype(str) + "-" + df["Bulan"].astype(str) + "-" + df["Tanggal"].astype(str))

    df = df.sort_values("Tanggal_Full")

    df["rain_prev_1"] = df["Curah_Hujan"].shift(1)
    df["rain_prev_3"] = df["Curah_Hujan"].rolling(3).mean()
    df["rain_prev_7"] = df["Curah_Hujan"].rolling(7).mean()
    df["rain_prev_14"] = df["Curah_Hujan"].rolling(14).sum()
    df["rain_prev_30"] = df["Curah_Hujan"].rolling(30).sum()

    df = df.dropna()

    df["Label"] = (df["Curah_Hujan"] >= 1).astype(int)

    df_major = df[df["Label"] == 0]
    df_minor = df[df["Label"] == 1]
    df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
    df_bal = pd.concat([df_major, df_minor_up])

    df_bal["Kabupaten_Code"] = df_bal["Kabupaten"].astype("category").cat.codes

    df_bal["sin_bulan"] = np.sin(2 * np.pi * df_bal["Bulan"] / 12)
    df_bal["cos_bulan"] = np.cos(2 * np.pi * df_bal["Bulan"] / 12)
    df_bal["sin_tgl"] = np.sin(2 * np.pi * df_bal["Tanggal"] / 31)
    df_bal["cos_tgl"] = np.cos(2 * np.pi * df_bal["Tanggal"] / 31)

    df_bal["Musim"] = df_bal["Bulan"].apply(lambda x: 1 if x in [10, 11, 12, 1, 2, 3] else 0)

    fitur = df_bal[[
        "Kabupaten_Code",
        "sin_bulan", "cos_bulan",
        "sin_tgl", "cos_tgl",
        "Musim",
        "rain_prev_1", "rain_prev_3", "rain_prev_7",
        "rain_prev_14", "rain_prev_30"
    ]]

    y = df_bal["Label"]

    X_train, X_test, y_train, y_test = train_test_split(fitur, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc*100:.2f}%")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(models_path, "model_rf.pkl"))
    joblib.dump({
        "kabupaten_mapping": df_bal["Kabupaten"].astype("category").cat.categories.tolist()
    }, os.path.join(models_path, "encoder_kabupaten.pkl"))

    print("Model berhasil disimpan.")

if __name__ == "__main__":
    train_model()
