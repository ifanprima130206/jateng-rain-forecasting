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

def extract_metadata(text):
    meta = {}
    p = {
        "Nama Pos": r"(Nama Pos|Pos)\s*[: ]+\s*(.+)",
        "Kabupaten": r"(Kabupaten|Kota/Kabupaten)\s*[: ]+\s*(.+)",
        "Kecamatan": r"(Kecamatan)\s*[: ]+\s*(.+)"
    }
    for key, pattern in p.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            meta[key] = m.group(2).strip()
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
    print(f"Memproses: {pdf_path}")

    all_rows = []
    last_meta = {"Nama Pos": "Unknown", "Kabupaten": "Unknown", "Kecamatan": "Unknown"}

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            print(f"Page {idx}/{len(pdf.pages)}", end="\r")

            text = page.extract_text() or ""
            meta = extract_metadata(text)

            if meta:
                last_meta.update(meta)

            for line in text.splitlines():
                parsed = parse_table_line(line)
                if not parsed:
                    continue

                day, nums = parsed
                all_rows.append({
                    "Nama Pos": last_meta["Nama Pos"],
                    "Kabupaten": last_meta["Kabupaten"],
                    "Kecamatan": last_meta["Kecamatan"],
                    "Tanggal": day,
                    "Jan": nums[0], "Feb": nums[1], "Mar": nums[2],
                    "Apr": nums[3], "Mei": nums[4], "Jun": nums[5],
                    "Jul": nums[6], "Ags": nums[7], "Sep": nums[8],
                    "Okt": nums[9], "Nov": nums[10], "Des": nums[11]
                })

    df = pd.DataFrame(all_rows)
    print(f"\nSelesai. Total data: {len(df)}")
    return df

def save_output(df, output_folder, filename):
    if df.empty:
        print("Tidak ada data terekstrak.")
        return

    file_path = os.path.join(output_folder, filename + ".csv")
    df.to_csv(file_path, index=False)
    print(f"File berhasil disimpan: {file_path}")

if __name__ == "__main__":
    pdf_path = input("Masukkan path file PDF: ").strip()
    output_folder = input("Masukkan folder output: ").strip()
    output_name = input("Masukkan nama file output (tanpa .csv): ").strip()

    if not os.path.exists(pdf_path):
        print("PDF tidak ditemukan.")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = process_pdf(pdf_path)
    save_output(df, output_folder, output_name)
