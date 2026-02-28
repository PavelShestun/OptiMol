from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
import numpy as np

def calculate_cns_mpo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return 0
    
    # CNS MPO включает 6 параметров. Вот упрощенная, но научно обоснованная версия:
    # 1. LogP (от 0 до 5, идеал < 3)
    logp = Descriptors.MolLogP(mol)
    w_logp = np.interp(logp, [0, 3, 5], [1, 1, 0])
    
    # 2. MW (Molecular Weight, идеал < 360)
    mw = Descriptors.MolWt(mol)
    w_mw = np.interp(mw, [200, 360, 500], [1, 1, 0])
    
    # 3. TPSA (Polar Surface Area, идеал 40-90)
    tpsa = Descriptors.TPSA(mol)
    w_tpsa = np.interp(tpsa, [20, 40, 90, 140], [0, 1, 1, 0])
    
    # 4. HBD (Hydrogen Bond Donors, идеал < 3)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    w_hbd = np.interp(hbd, [0, 3, 5], [1, 1, 0])
    
    return (w_logp + w_mw + w_tpsa + w_hbd) * (6.0 / 4.0) # Масштабируем до 6

def get_full_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return {
        'smiles': smiles,
        'qed': QED.qed(mol),
        'cns_mpo': calculate_cns_mpo(smiles),
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_atoms': mol.GetNumAtoms()
    }
