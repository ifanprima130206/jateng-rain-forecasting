import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px

# Configuration
st.set_page_config(page_title="Jateng Rain Forecast", page_icon="🌧️", layout="wide")

# Constants & Paths
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'modelling', 'saved_models')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset', 'processed', 'data_training_gabungan.csv')

# Load Data & Models
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(os.path.join(MODELS_PATH, 'model_rf.pkl'))
        encoder = joblib.load(os.path.join(MODELS_PATH, 'encoder_kabupaten.pkl'))
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        if 'Curah Hujan' in df.columns: df.rename(columns={'Curah Hujan': 'Curah_Hujan'}, inplace=True)
        return model, encoder, df.sort_values('Date')
    except Exception as e:
        return None, None, None

model, encoder_data, df_data = load_resources()

# Helper Functions
def get_historical_features(kabupaten, date, df):
    df_kab = df[df['Kabupaten'] == kabupaten]
    target_date = pd.to_datetime(date)
    
    if target_date > df_kab['Date'].max():
        return _get_average_features(df_kab, target_date.month)
        
    loc = df_kab[df_kab['Date'] == target_date]
    if loc.empty: return _get_average_features(df_kab, target_date.month)
    
    rain_vals = df_kab[df_kab['Date'] < target_date].tail(30)['Curah_Hujan'].values
    if len(rain_vals) < 1: return _get_average_features(df_kab, target_date.month)

    return {
        'rain_prev_1': rain_vals[-1],
        'rain_prev_3': rain_vals[-3:].mean(),
        'rain_prev_7': rain_vals[-7:].mean(),
        'rain_prev_14': rain_vals[-14:].sum(),
        'rain_prev_30': rain_vals[-30:].sum()
    }

def _get_average_features(df_kab, month):
    avg_rain = df_kab[df_kab['Bulan'] == month]['Curah_Hujan'].mean()
    avg_rain = 0 if pd.isna(avg_rain) else avg_rain
    return {
        'rain_prev_1': avg_rain, 'rain_prev_3': avg_rain, 'rain_prev_7': avg_rain,
        'rain_prev_14': avg_rain * 14, 'rain_prev_30': avg_rain * 30
    }

# UI & Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Menu", ["🏠 Prediksi", "📊 EDA"])

if page == "🏠 Prediksi":
    st.title("🌧️ Jateng Rain Forecast")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Parameter")
        kab_list = encoder_data['kabupaten_mapping'] if encoder_data else []
        selected_kab = st.selectbox("Wilayah", kab_list)
        selected_date = st.date_input("Tanggal", datetime.today())
        
        if st.button("Mulai Prediksi", type="primary"):
            date_obj = pd.to_datetime(selected_date)
            # Feature Engineering
            try: kab_code = kab_list.index(selected_kab)
            except: kab_code = -1
            
            month, day = date_obj.month, date_obj.day
            hist_feat = get_historical_features(selected_kab, date_obj, df_data)
            
            input_data = pd.DataFrame([{
                "Kabupaten_Code": kab_code,
                "sin_bulan": np.sin(2 * np.pi * month / 12), "cos_bulan": np.cos(2 * np.pi * month / 12),
                "sin_tgl": np.sin(2 * np.pi * day / 31), "cos_tgl": np.cos(2 * np.pi * day / 31),
                "Musim": 1 if month in [10, 11, 12, 1, 2, 3] else 0,
                **hist_feat
            }])
            
            # Prediction
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            with col2:
                st.markdown("---")
                st.subheader("Hasil Prediksi Pada Tanggal Yang Dipilih")
                if pred == 1:
                    st.error(f"🌧️ HUJAN (Probabilitas: {prob:.1%})")
                    st.caption("Sediakan payung/jas hujan.")
                else:
                    st.success(f"☀️ CERAH/BERAWAN (Probabilitas Hujan: {prob:.1%})")
                    st.caption("Aman untuk aktivitas luar.")
                
                with st.expander("Detail Input Features"):
                    st.json(hist_feat)

                # 5-Day Forecast
                st.markdown("---")
                st.subheader("📅 Ramalan 5 Hari ke Depan")
                
                forecast_cols = st.columns(5)
                
                for i in range(1, 6):
                    next_date = date_obj + timedelta(days=i)
                    n_month, n_day = next_date.month, next_date.day
                    n_hist_feat = get_historical_features(selected_kab, next_date, df_data)
                    
                    n_input = pd.DataFrame([{
                        "Kabupaten_Code": kab_code,
                        "sin_bulan": np.sin(2 * np.pi * n_month / 12), "cos_bulan": np.cos(2 * np.pi * n_month / 12),
                        "sin_tgl": np.sin(2 * np.pi * n_day / 31), "cos_tgl": np.cos(2 * np.pi * n_day / 31),
                        "Musim": 1 if n_month in [10, 11, 12, 1, 2, 3] else 0,
                        **n_hist_feat
                    }])
                    
                    n_pred = model.predict(n_input)[0]
                    n_prob = model.predict_proba(n_input)[0][1]
                    
                    with forecast_cols[i-1]:
                        st.markdown(f"**{next_date.strftime('%d/%m')}**")
                        if n_pred == 1:
                             st.markdown("🌧️ **Hujan**")
                             st.progress(int(n_prob*100))
                        else:
                             st.markdown("☀️ **Cerah**")
                             st.progress(int(n_prob*100))
                        st.caption(f"{n_prob*100:.0f}%")

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    
    if df_data is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data", len(df_data))
        c2.metric("Wilayah", df_data['Kabupaten'].nunique())
        c3.metric("Periode", f"{df_data['Date'].min():%Y-%m} s/d {df_data['Date'].max():%Y-%m}")
        
        st.markdown("---")
        
        # 1. Line Plots
        st.subheader("1. Analisis Tren (Line Plot)")
        st.caption("Grafik garis digunakan untuk melihat pola perubahan curah hujan berdasarkan waktu.")
        tab_lp1, tab_lp2 = st.tabs(["Pola Musiman (Bulanan)", "Tren Tahunan"])
        
        with tab_lp1:
            avg_month = df_data.groupby('Bulan')['Curah_Hujan'].mean().reset_index()
            fig_lp1 = px.line(avg_month, x='Bulan', y='Curah_Hujan', markers=True, 
                             title='Rata-rata Curah Hujan per Bulan')
            st.plotly_chart(fig_lp1, use_container_width=True)
            st.info("Puncak curah hujan tertinggi biasanya terjadi di awal dan akhir tahun (Januari - Maret, Oktober - Desember).")

        with tab_lp2:
            avg_year = df_data.groupby('Tahun')['Curah_Hujan'].mean().reset_index()
            fig_lp2 = px.line(avg_year, x='Tahun', y='Curah_Hujan', markers=True, 
                             title='Rata-rata Curah Hujan per Tahun')
            st.plotly_chart(fig_lp2, use_container_width=True)
            st.info("Grafik ini menunjukkan fluktuasi rata-rata intensitas hujan dari tahun ke tahun.")

        st.markdown("---")
        
        # 2. Box Plots
        st.subheader("2. Distribusi Data (Box Plot)")
        st.caption("Box plot berguna untuk melihat sebaran data dan mendeteksi outlier (nilai ekstrem).")
        tab_bp1, tab_bp2 = st.tabs(["Sebaran per Bulan", "Sebaran per Wilayah (Top 10)"])
        
        with tab_bp1:
            fig_bp1 = px.box(df_data, x='Bulan', y='Curah_Hujan', title='Distribusi Curah Hujan per Bulan')
            st.plotly_chart(fig_bp1, use_container_width=True)
            st.info("Box plot ini memperlihatkan variasi curah hujan di setiap bulan. Kotak yang lebih panjang menandakan variasi yang lebih besar.")
            
        with tab_bp2:
            top_kab = df_data.groupby('Kabupaten')['Curah_Hujan'].mean().nlargest(10).index
            df_top = df_data[df_data['Kabupaten'].isin(top_kab)]
            fig_bp2 = px.box(df_top, x='Kabupaten', y='Curah_Hujan', title='Distribusi Curah Hujan di 10 Wilayah Terbasah')
            st.plotly_chart(fig_bp2, use_container_width=True)
            st.info("Distribusi ini fokus pada 10 wilayah dengan rata-rata hujan tertinggi.")

        st.markdown("---")

        # 3. Pie Charts
        st.subheader("3. Proporsi Data (Pie Chart)")
        st.caption("Pie chart menunjukkan persentase atau bagian dari keseluruhan.")
        tab_pc1, tab_pc2 = st.tabs(["Proporsi Label", "Kontribusi Hujan per Bulan"])
        
        with tab_pc1:
            if 'Label' in df_data.columns:
                fig_pc1 = px.pie(df_data, names='Label', title='Persentase Hari Hujan (1) vs Tidak (0)',
                                color_discrete_sequence=['#FFD200', '#4facfe'])
                st.plotly_chart(fig_pc1, use_container_width=True)
                st.info("Menunjukkan seberapa sering hujan terjadi dibandingkan hari cerah dalam dataset.")
        
        with tab_pc2:
            # Filter only rainy days
            rainy_days = df_data[df_data['Curah_Hujan'] > 0]
            rain_counts = rainy_days['Bulan'].value_counts().reset_index()
            rain_counts.columns = ['Bulan', 'Kejadian']
            fig_pc2 = px.pie(rain_counts, values='Kejadian', names='Bulan', title='Proporsi Kejadian Hujan Berdasarkan Bulan')
            st.plotly_chart(fig_pc2, use_container_width=True)
            st.info("Bulan mana yang paling sering menyumbang kejadian hujan? Pie chart ini membagi total kejadian hujan berdasarkan bulan.")

        st.markdown("---")

        # 4. Scatter Plot
        st.subheader("4. Hubungan Antar Variabel (Scatter Plot)")
        st.caption("Scatter plot digunakan untuk melihat korelasi atau pola persebaran antara dua variabel.")
        
        # Sample data to avoid overplotting if dataset is huge
        sample_df = df_data.sample(min(5000, len(df_data)), random_state=42)
        fig_sp = px.scatter(sample_df, x='Bulan', y='Curah_Hujan', color='Curah_Hujan', 
                           title='Scatter Plot: Bulan vs Intensitas Hujan (Sample 5000 Data)',
                           color_continuous_scale='Bluered')
        st.plotly_chart(fig_sp, use_container_width=True)
        st.info("Plot ini memperlihatkan sebaran intensitas hujan di setiap bulan. Titik-titik yang lebih tinggi menunjukkan hari dengan hujan sangat lebat.")

        st.markdown("---")

        # 5. Correlation Matrix
        st.subheader("5. Matriks Korelasi")
        st.caption("Heatmap korelasi menunjukkan seberapa kuat hubungan antar fitur numerik.")
        
        numeric_cols = ['Curah_Hujan', 'Bulan', 'Tahun', 'Tanggal']
        if 'Label' in df_data.columns: numeric_cols.append('Label')
        
        corr_matrix = df_data[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', 
                            title='Korelasi Antar Fitur Numerik', origin='lower')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("Angka mendekati 1 berarti korelasi positif kuat, -1 korelasi negatif kuat, dan 0 tidak ada hubungan linier.")
