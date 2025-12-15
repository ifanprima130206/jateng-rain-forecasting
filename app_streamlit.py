import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Jateng Rain Forecast",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #1a1a1a;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 201, 255, 0.4);
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(CURRENT_DIR, 'app', 'modelling', 'saved_models')
DATA_PATH = os.path.join(CURRENT_DIR, 'Dataset', 'processed', 'data_training_gabungan.csv')

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(os.path.join(MODELS_PATH, 'model_rf.pkl'))
        encoder_data = joblib.load(os.path.join(MODELS_PATH, 'encoder_kabupaten.pkl'))
        
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        return model, encoder_data, df
    except Exception as e:
        return None, None, None

model, encoder_data, df_data = load_resources()

def get_historical_features(kabupaten, date, df):
    df_kab = df[df['Kabupaten'] == kabupaten].copy()
    
    target_date = pd.to_datetime(date)
    
    max_date = df_kab['Date'].max()
    
    features = {}
    lags = [1, 3, 7, 14, 30]
    
    if target_date <= max_date:
        loc = df_kab[df_kab['Date'] == target_date]
        if not loc.empty:
            curr_idx = loc.index[0]
            
            past_30 = df_kab[df_kab['Date'] < target_date].tail(30)
            
            rain_vals = past_30['Curah_Hujan'].values
            
            features['rain_prev_1'] = rain_vals[-1] if len(rain_vals) >= 1 else 0
            features['rain_prev_3'] = rain_vals[-3:].mean() if len(rain_vals) >= 1 else 0
            features['rain_prev_7'] = rain_vals[-7:].mean() if len(rain_vals) >= 1 else 0
            features['rain_prev_14'] = rain_vals[-14:].sum() if len(rain_vals) >= 1 else 0
            features['rain_prev_30'] = rain_vals[-30:].sum() if len(rain_vals) >= 1 else 0
            
        else:
             features = _get_average_features(df_kab, target_date.month)
    else:
        features = _get_average_features(df_kab, target_date.month)
        
    return features

def _get_average_features(df_kab, month):
    df_month = df_kab[df_kab['Bulan'] == month]
    
    if df_month.empty:
        return {
            'rain_prev_1': 0, 'rain_prev_3': 0, 'rain_prev_7': 0,
            'rain_prev_14': 0, 'rain_prev_30': 0
        }

    avg_rain = df_month['Curah_Hujan'].mean()
    
    if pd.isna(avg_rain):
        avg_rain = 0
        
    return {
        'rain_prev_1': avg_rain,
        'rain_prev_3': avg_rain,
        'rain_prev_7': avg_rain,
        'rain_prev_14': avg_rain * 14,
        'rain_prev_30': avg_rain * 30
    }

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414974.png", width=100)
    st.title("Jateng Rain Forecast")
    st.markdown("Prediksi Curah Hujan Harian berbasis *Random Forest*.")
    
    if model is None:
        st.error("Model tidak ditemukan! Pastikan model telah dilatih.")
        st.stop()
        
    st.markdown("### ⚙️ Parameter")
    
    kabupaten_list = encoder_data['kabupaten_mapping']
    selected_kab = st.selectbox("Pilih Kabupaten/Kota", kabupaten_list)
    
    selected_date = st.date_input("Pilih Tanggal", datetime.today())
    
    st.info(f"Memprediksi untuk wilayah **{selected_kab}** pada tanggal **{selected_date.strftime('%d %B %Y')}**.")
    
    predict_btn = st.button("🌦️ Mulai Prediksi")

with col2:
    if predict_btn:
        with st.spinner("Menganalisis data atmosfer..."):
            date_obj = pd.to_datetime(selected_date)
            
            try:
                kab_code = kabupaten_list.index(selected_kab)
            except ValueError:
                kab_code = -1
            
            bulan = date_obj.month
            tanggal = date_obj.day
            
            sin_bulan = np.sin(2 * np.pi * bulan / 12)
            cos_bulan = np.cos(2 * np.pi * bulan / 12)
            sin_tgl = np.sin(2 * np.pi * tanggal / 31)
            cos_tgl = np.cos(2 * np.pi * tanggal / 31)
            
            musim = 1 if bulan in [10, 11, 12, 1, 2, 3] else 0
            
            hist_feat = get_historical_features(selected_kab, date_obj, df_data)
            
            input_data = pd.DataFrame([{
                "Kabupaten_Code": kab_code,
                "sin_bulan": sin_bulan,
                "cos_bulan": cos_bulan,
                "sin_tgl": sin_tgl,
                "cos_tgl": cos_tgl,
                "Musim": musim,
                "rain_prev_1": hist_feat['rain_prev_1'],
                "rain_prev_3": hist_feat['rain_prev_3'],
                "rain_prev_7": hist_feat['rain_prev_7'],
                "rain_prev_14": hist_feat['rain_prev_14'],
                "rain_prev_30": hist_feat['rain_prev_30']
            }])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("### Prediksi Curah Hujan")
                if prediction == 1:
                    st.markdown("<h2 style='color: #4facfe;'>🌧️ Hujan Terdeteksi</h2>", unsafe_allow_html=True)
                    st.markdown(f"Peluang Hujan: **{probability*100:.1f}%**")
                    st.markdown("Disarankan membawa payung atau jas hujan.")
                else:
                    st.markdown("<h2 style='color: #FFD200;'>☀️ Cerah / Berawan</h2>", unsafe_allow_html=True)
                    st.markdown(f"Peluang Hujan: **{probability*100:.1f}%**")
                    st.markdown("Cuaca diprediksi aman untuk aktivitas luar ruangan.")
            
            with res_col2:
                with st.expander("Lihat Data Input"):
                    st.write("Data Historis (Estimasi):")
                    st.json(hist_feat)

            st.markdown("---")
            st.markdown("### 📅 Ramalan 5 Hari ke Depan")
            
            forecast_cols = st.columns(5)
            
            for i in range(1, 6):
                next_date = date_obj + timedelta(days=i)
                
                n_bulan = next_date.month
                n_tanggal = next_date.day
                
                n_sin_bulan = np.sin(2 * np.pi * n_bulan / 12)
                n_cos_bulan = np.cos(2 * np.pi * n_bulan / 12)
                n_sin_tgl = np.sin(2 * np.pi * n_tanggal / 31)
                n_cos_tgl = np.cos(2 * np.pi * n_tanggal / 31)
                
                n_musim = 1 if n_bulan in [10, 11, 12, 1, 2, 3] else 0
                
                n_hist_feat = get_historical_features(selected_kab, next_date, df_data)
                
                n_input_data = pd.DataFrame([{
                    "Kabupaten_Code": kab_code,
                    "sin_bulan": n_sin_bulan,
                    "cos_bulan": n_cos_bulan,
                    "sin_tgl": n_sin_tgl,
                    "cos_tgl": n_cos_tgl,
                    "Musim": n_musim,
                    "rain_prev_1": n_hist_feat['rain_prev_1'],
                    "rain_prev_3": n_hist_feat['rain_prev_3'],
                    "rain_prev_7": n_hist_feat['rain_prev_7'],
                    "rain_prev_14": n_hist_feat['rain_prev_14'],
                    "rain_prev_30": n_hist_feat['rain_prev_30']
                }])
                
                n_pred = model.predict(n_input_data)[0]
                n_prob = model.predict_proba(n_input_data)[0][1]
                
                with forecast_cols[i-1]:
                    st.markdown(f"**{next_date.strftime('%d/%m')}**")
                    if n_pred == 1:
                         st.markdown("🌧️ **Hujan**")
                         st.progress(int(n_prob*100))
                    else:
                         st.markdown("☀️ **Cerah**")
                         st.progress(int(n_prob*100))
                    
                    st.caption(f"{n_prob*100:.0f}%")

    else:
        st.markdown("""
        ### Selamat Datang
        Aplikasi ini menggunakan Machine Learning untuk memprediksi kemungkinan hujan di wilayah Jawa Tengah.
        
        **Cara Menggunakan:**
        1. Pilih Kabupaten/Kota di panel kiri.
        2. Tentukan Tanggal.
        3. Klik tombol **Mulai Prediksi**.
        """)
        st.image("https://images.unsplash.com/photo-1601134467661-3d775b999c8b?q=80&w=1075&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Jawa Tengah Rain Forecast", use_container_width=True)
