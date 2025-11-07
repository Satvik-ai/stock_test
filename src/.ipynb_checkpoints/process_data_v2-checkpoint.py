import pandas as pd
from pathlib import Path

# Define paths consistent with the project structure
RAW_DATA_PATH = Path("raw_data/v2")
PROCESSED_DATA_PATH = Path("data")
MERGED_PARQUET_PATH = PROCESSED_DATA_PATH / "stock_data_full.parquet"
SAMPLED_PARQUET_PATH = PROCESSED_DATA_PATH / "stock_data.parquet"
SAMPLE_SIZE = 1000 # Define a sample size for training

def clean_filename(filename: str) -> str:
    """Extracts the stock ticker from the raw filename."""
    return filename.split("__")[0]

def process_and_engineer_features(file_path: Path):
    """
    Reads a raw CSV file, cleans it, engineers features suitable for
    minute-level data, and returns a processed DataFrame.
    """
    print(f"Processing: {file_path.name}")
    df = pd.read_csv(file_path)

    # --- Data Cleaning and Typing ---
    df['event_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.drop(columns=['timestamp'])
    stock_id = clean_filename(file_path.name)
    df["stock_v2_id"] = [f"{stock_id}-v2-{i}" for i in df.index]
    
    # Sort by timestamp before doing any window-based calculations
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    # --- Feature Engineering ---
    df['ma_15_min'] = df['close'].rolling(window=15).mean()
    df['ma_60_min'] = df['close'].rolling(window=60).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # --- Final Selection and Cleanup ---
    final_cols = [
        'event_timestamp', 'stock_v2_id', 'open', 'high', 'low', 'close', 
        'volume', 'ma_15_min', 'ma_60_min', 'rsi_14', 'target'
    ]
    df_processed = df[final_cols]
    
    return df_processed.dropna()

def main():
    """
    Orchestrates the data processing workflow: processes raw files, merges them,
    saves the full dataset, and creates a smaller sampled dataset for training.
    """
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for file in RAW_DATA_PATH.glob("*.csv"):
        try:
            processed_df = process_and_engineer_features(file)
            all_dfs.append(processed_df)
        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df = merged_df.sort_values(by="event_timestamp").reset_index(drop=True)
        
        # Save the full processed dataset
        merged_df.to_parquet(MERGED_PARQUET_PATH, index=False)
        print(f"\n✅ Full dataset saved to: {MERGED_PARQUET_PATH}")
        print(f"Total rows: {len(merged_df)}")

        # Create and save a smaller, sampled dataset for training
        if len(merged_df) > SAMPLE_SIZE:
            sampled_df = merged_df.sample(n=SAMPLE_SIZE, random_state=42)
            sampled_df.to_parquet(SAMPLED_PARQUET_PATH, index=False)
            print(f"✅ Sampled dataset ({SAMPLE_SIZE} rows) saved to: {SAMPLED_PARQUET_PATH}")
        else:
            # If the dataset is small, the sampled file is just a copy
            merged_df.to_parquet(SAMPLED_PARQUET_PATH, index=False)
            print(f"✅ Dataset is smaller than sample size. Full data saved to: {SAMPLED_PARQUET_PATH}")

    else:
        print("❌ No data to merge.")

if __name__ == "__main__":
    main()