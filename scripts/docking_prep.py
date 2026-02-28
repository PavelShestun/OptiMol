from rdkit import Chem
from rdkit.Chem import AllChem

def prepare_ligand(smiles, output_pdbqt):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    Chem.MolToPDBFile(mol, "temp_ligand.pdb")
    # Для SOTA нужно конвертировать в pdbqt, но GNINA ест и PDB/SDF
    return "temp_ligand.pdb"

print("Docking prep ready.")
