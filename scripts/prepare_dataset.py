from optimol.utils.curation import process_raw_data
import os

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    process_raw_data("data/raw/keap1_raw.csv", "data/processed/keap1_cleaned.csv")
