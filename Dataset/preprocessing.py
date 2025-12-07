import os
import pandas as pd
import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_data_path = os.path.join(project_root, 'dataset', 'output')
processed_path = os.path.join(project_root, 'dataset', 'processed')

os.makedirs(processed_path, exist_ok=True)

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Mei': 5, 'Jun': 6,
    'Jul': 7, 'Ags': 8, 'Sep': 9, 'Okt': 10, 'Nov': 11, 'Des': 12
}

def process_data():
    all_data = []
    
    years = range(2019, 2025) 
    
    print("Mulai memproses data...")
    
    for year in years:
        year_path = os.path.join(raw_data_path, str(year))
        
        if not os.path.exists(year_path):
            print(f"Folder tahun {year} tidak ditemukan, skip.")
            continue
            
        csv_files = glob.glob(os.path.join(year_path, "*.csv"))
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                df['Tahun'] = year 
                
                id_vars = ['Nama Pos', 'Kabupaten', 'Kecamatan', 'Tanggal', 'Tahun']
                
                value_vars = [col for col in month_map.keys() if col in df.columns]
                
                if not value_vars:
                    print(f"Warning: Tidak ada kolom bulan di file {os.path.basename(file_path)}")
                    continue

                df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, 
                                    var_name='Bulan_Str', value_name='Curah_Hujan')
                
                df_melted['Bulan'] = df_melted['Bulan_Str'].map(month_map)
                
                all_data.append(df_melted)
                
            except Exception as e:
                print(f"Error memproses file {file_path}: {e}")

    if not all_data:
        print("Tidak ada data yang berhasil diproses.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    
    final_df['Date_Str'] = final_df['Tahun'].astype(str) + '-' + \
                           final_df['Bulan'].astype(str) + '-' + \
                           final_df['Tanggal'].astype(str)
    
    final_df['Date'] = pd.to_datetime(final_df['Date_Str'], format='%Y-%m-%d', errors='coerce')
    
    final_df = final_df.dropna(subset=['Date', 'Curah_Hujan'])
    
    final_df['Label'] = final_df['Curah_Hujan'].apply(lambda x: 1 if x >= 1 else 0)
    
    cols = ['Date', 'Tahun', 'Bulan', 'Tanggal', 'Nama Pos', 'Kabupaten', 'Curah_Hujan', 'Label']
    final_df = final_df[cols].sort_values(by=['Nama Pos', 'Date'])
    
    output_file = os.path.join(processed_path, 'data_training_gabungan.csv')
    final_df.to_csv(output_file, index=False)
    
    print(f"Selesai! Data tersimpan di: {output_file}")
    print(f"Total baris data: {len(final_df)}")
    print("Contoh 5 data teratas:")
    print(final_df.head())

if __name__ == "__main__":
    process_data()