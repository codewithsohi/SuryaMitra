import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

PHYSICAL_COLS = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", 
    "SAVNCPP", "USFLUX", "TOTFZ", "MEANPOT", "R_VALUE", 
    "AREA_ACR", "MEANGAM", "MEANGBZ", "MEANGBT"
]

def extract_label_from_path(filepath):
    """
    Extracts binary label: 1 if >= M-class flare, 0 otherwise.
    """
    parent_folder = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    
    # Non-flaring active regions are always negative
    if parent_folder.upper() == "NF":
        return 0
    
    # In FL folder: M and X flares are positive (1), B and C are negative (0)
    prefix = filename.split("@")[0].upper()
    if prefix.startswith(("M", "X")):
        return 1
    return 0

def process_single_file(filepath):
    try:
        df = pd.read_csv(filepath, sep="\t")
    except Exception:
        return None

    if df.empty:
        return None

    record = {
        "file_id": os.path.basename(filepath),
        "label": extract_label_from_path(filepath)
    }

    # Extract statistical aggregates over the 12-hour observation window
    for col in PHYSICAL_COLS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                record[f"{col}_mean"] = series.mean()
                record[f"{col}_std"] = series.std(ddof=0)
                record[f"{col}_min"] = series.min()
                record[f"{col}_max"] = series.max()
                record[f"{col}_delta"] = series.iloc[-1] - series.iloc[0]
            else:
                record[f"{col}_mean"] = np.nan
                record[f"{col}_std"] = np.nan
                record[f"{col}_min"] = np.nan
                record[f"{col}_max"] = np.nan
                record[f"{col}_delta"] = np.nan

    return record

def process_partition(input_dir, output_csv):
    files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    print(f"Processing {len(files)} files from {input_dir}...")
    
    rows = []
    for f in tqdm(files):
        row = process_single_file(f)
        if row is not None:
            rows.append(row)
            
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved feature table to: {output_csv}")
    print(f"Shape: {df_out.shape}")
    print("\nClass Distribution:")
    print(df_out["label"].value_counts())
    print("\nPercentage breakdown:")
    print(df_out["label"].value_counts(normalize=True) * 100)
    return df_out

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    process_partition("data/raw/Partition1", "data/processed/partition1_tabular.csv")