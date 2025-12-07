import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

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
    df['Hujan_Kemarin'] = df['Curah_Hujan'].shift(1)
    df = df.dropna()

    df_major = df[df['Label'] == 0]
    df_minor = df[df['Label'] == 1]
    df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
    df_bal = pd.concat([df_major, df_minor_up])

    fitur = df_bal[['Bulan', 'Tanggal', 'Kabupaten', 'Hujan_Kemarin']]
    y = df_bal['Label']

    encoder = OneHotEncoder(handle_unknown='ignore')
    kab_encoded = encoder.fit_transform(fitur[['Kabupaten']]).toarray()
    kab_cols = encoder.get_feature_names_out(['Kabupaten'])

    df_final = pd.DataFrame({
        "Bulan": fitur["Bulan"].values,
        "Tanggal": fitur["Tanggal"].values,
        "Hujan_Kemarin": fitur["Hujan_Kemarin"].values
    })

    df_ohe = pd.DataFrame(kab_encoded, columns=kab_cols)
    X = pd.concat([df_final, df_ohe], axis=1)

    joblib.dump(list(X.columns), os.path.join(models_path, 'feature_columns.pkl'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc*100:.2f}%")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(models_path, 'model_rf.pkl'))
    joblib.dump(encoder, os.path.join(models_path, 'encoder_kabupaten.pkl'))

    print("Model dan encoder berhasil disimpan.")

if __name__ == "__main__":
    train_model()
