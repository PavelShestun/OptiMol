import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from loguru import logger

def curate_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # Удаляем соли (оставляем только самый большой фрагмент)
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda x: x.GetNumAtoms())
        # Канонизация SMILES
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except:
        return None

def process_raw_data(input_path, output_path):
    df = pd.read_csv(input_path)
    initial_count = len(df)
    
    # 1. Оставляем только нужные колонки и удаляем пустые значения
    df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']]
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    
    # 2. Приводим всё к nM и фильтруем только nM
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[df['standard_units'] == 'nM']
    
    # 3. Очистка структур
    logger.info("Curating chemical structures...")
    df['canonical_smiles'] = df['canonical_smiles'].apply(curate_molecule)
    df = df.dropna(subset=['canonical_smiles'])
    
    # 4. Агрегация дубликатов (берем медиану IC50 для одинаковых SMILES)
    df = df.groupby('canonical_smiles')['standard_value'].median().reset_index()
    
    # 5. Расчет pIC50
    # pIC50 = -log10(IC50 M) = -log10(IC50 nM * 1e-9) = 9 - log10(IC50 nM)
    df['pIC50'] = 9 - np.log10(df['standard_value'] + 1e-9)
    
    # 6. Удаление выбросов (например, слишком неактивные)
    df = df[df['pIC50'] > 2] # Ниже 100uM — обычно шум
    
    logger.success(f"Curation finished: {initial_count} -> {len(df)} unique molecules")
    df.to_csv(output_path, index=False)
    return df
