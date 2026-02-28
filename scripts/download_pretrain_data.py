import pandas as pd
from chembl_webresource_client.new_client import new_client
from loguru import logger
import os

def download_general_molecules(limit=10000):
    os.makedirs("data/raw", exist_ok=True)
    logger.info(f"Downloading {limit} molecules from ChEMBL...")
    
    mol_client = new_client.molecule
    # Фильтруем: вес до 500, только те, у кого есть структура
    res = mol_client.filter(molecule_properties__mw_freebase__lte=500) \
                    .filter(molecule_structures__isnull=False) \
                    .only(['molecule_structures'])[:limit]
    
    smiles_list = []
    for entry in res:
        try:
            # Извлекаем SMILES из вложенного словаря
            smiles = entry['molecule_structures']['canonical_smiles']
            if smiles:
                smiles_list.append(smiles)
        except (KeyError, TypeError):
            continue
            
    df = pd.DataFrame(smiles_list, columns=['canonical_smiles'])
    df = df.drop_duplicates()
    
    save_path = "data/raw/pretrain_data.csv"
    df.to_csv(save_path, index=False)
    logger.success(f"Successfully saved {len(df)} unique molecules to {save_path}")

if __name__ == "__main__":
    download_general_molecules()
