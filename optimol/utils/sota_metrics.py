from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
import warnings

# Отключаем предупреждения RDKit для чистоты консоли
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_sa_score(mol):
    if not mol: return 10
    num_rings = mol.GetRingInfo().NumRings()
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    # Упрощенная формула SA
    score = 1 + (0.5 * num_rings) + (2 * num_spiro) + (2 * num_bridge)
    return min(score, 10)

def get_diversity_penalty(smiles, buffer_smiles):
    if not buffer_smiles: return 1.0
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return 0.0
    
    # Новый синтаксис RDKit для Morgan Fingerprints
    gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=1024)
    fp = gen.GetFingerprint(mol)
    
    max_sim = 0
    for b_smi in buffer_smiles:
        b_mol = Chem.MolFromSmiles(b_smi)
        if not b_mol: continue
        b_fp = gen.GetFingerprint(b_mol)
        sim = DataStructs.TanimotoSimilarity(fp, b_fp)
        max_sim = max(max_sim, sim)
    
    if max_sim > 0.8: return 0.2 # Штраф за слишком похожие молекулы
    return 1.0
