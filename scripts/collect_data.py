import os
import pandas as pd
from chembl_webresource_client.new_client import new_client
from loguru import logger

def download_keap1_data():
    # Создаем папку, если её нет
    os.makedirs("data/raw", exist_ok=True)
    
    logger.info("Connecting to ChEMBL...")
    target = new_client.target
    activity = new_client.activity
    
    # Поиск по ID мишени (KEAP1 Human)
    target_id = 'CHEMBL2069156' 
    
    logger.info(f"Fetching activities for target: {target_id}")
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    
    df = pd.DataFrame.from_dict(res)
    
    if df.empty:
        logger.error("No data found! Check the Target ID.")
        return
    
    save_path = "data/raw/keap1_raw.csv"
    df.to_csv(save_path, index=False)
    logger.success(f"Downloaded {len(df)} records and saved to {save_path}")
    
    # Выведем колонки, чтобы понимать, с чем работаем
    logger.info(f"Columns available: {list(df.columns)}")

if __name__ == "__main__":
    download_keap1_data()
