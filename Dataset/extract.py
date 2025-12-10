import pdfplumber
import pandas as pd
import re
import os

def clean_num(x):
    if x is None:
        return 0.0
    x = x.strip()
    if x in ["", "-", "."]:
        return 0.0
    x = x.replace(",", ".")
    x = re.sub(r"[^\d\.]", "", x)  
    try:
        return float(x)
    except:
        return 0.0

def extract_metadata_from_text(text):
    meta = {}
    patterns = {
        "Nama Pos": r"(Nama Pos|Pos)\s*[: ]+\s*(.+)",
        "Kabupaten": r"(Kabupaten|Kota/Kabupaten)\s*[: ]+\s*(.+)",
        "Kecamatan": r"(Kecamatan)\s*[: ]+\s*(.+)"
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            meta[key] = m.group(2).strip()
    return meta

def extract_metadata_from_table(page):
    meta = {}
    # GANTI: Gunakan extract_tables() (jamak) untuk mengambil SEMUA tabel di halaman,
    # termasuk tabel header kecil di atas.
    tables = page.extract_tables()
    
    if not tables:
        return meta

    for table in tables:
        for row in table:
            # Pastikan row memiliki data dan minimal 2 kolom
            # Filter None values agar tidak error saat di-strip()
            clean_row = [str(x).strip() if x is not None else "" for x in row]
            
            if len(clean_row) < 2:
                continue

            key = clean_row[0].lower() # Ubah ke huruf kecil biar gampang ceknya
            value = clean_row[1]

            # Lewati jika key kosong
            if not key:
                continue

            # LOGIKA PENCOCOKAN
            # Cek variasi "Nama Stasiun" atau "Nama Pos"
            if "nama stasiun" in key or "nama pos" in key or "stasiun" in key:
                # Ambil value, tapi jika value kosong, coba cek kolom berikutnya (jika ada)
                if value == "" and len(clean_row) > 2:
                    value = clean_row[2] # Kadang ada kolom kosong di tengah
                meta["Nama Pos"] = value
            
            elif "kabupaten" in key or "kota" in key:
                meta["Kabupaten"] = value
                
            elif "kecamatan" in key:
                meta["Kecamatan"] = value

    return meta

def parse_table_line(line):
    parts = line.strip().split()
    if len(parts) < 13:
        return None
    if not parts[0].isdigit():
        return None
    day = int(parts[0])
    if day < 1 or day > 31:
        return None
    nums = [clean_num(x) for x in parts[1:13]]
    return day, nums

def process_pdf(pdf_path):
    all_rows = []
    last_meta = {"Nama Pos": None, "Kabupaten": None, "Kecamatan": None}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            meta = extract_metadata_from_text(text)
            table_meta = extract_metadata_from_table(page)
            if table_meta:
                meta.update(table_meta)
            if meta:
                last_meta.update(meta)
            for line in text.splitlines():
                parsed = parse_table_line(line)
                if not parsed:
                    continue
                day, nums = parsed
                all_rows.append({
                    "Nama Pos": last_meta.get("Nama Pos") or "Unknown",
                    "Kabupaten": last_meta.get("Kabupaten") or "Unknown",
                    "Kecamatan": last_meta.get("Kecamatan") or "Unknown",
                    "Tanggal": day,
                    "Jan": nums[0], "Feb": nums[1], "Mar": nums[2],
                    "Apr": nums[3], "Mei": nums[4], "Jun": nums[5],
                    "Jul": nums[6], "Ags": nums[7], "Sep": nums[8],
                    "Okt": nums[9], "Nov": nums[10], "Des": nums[11]
                })
    return pd.DataFrame(all_rows)

def save_output(df, output_folder, filename):
    if df.empty:
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, filename + ".csv")
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    pdf_path = input("Masukkan path file PDF: ").strip()
    output_folder = input("Masukkan folder output: ").strip()
    output_name = input("Masukkan nama file output (tanpa .csv): ").strip()
    if not os.path.exists(pdf_path):
        exit()
    df = process_pdf(pdf_path)
    save_output(df, output_folder, output_name)
