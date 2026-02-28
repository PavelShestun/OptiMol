from rdkit import Chem
from rdkit.Chem import AllChem
import os

def make_3d(smiles, output_file):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("Invalid SMILES")
        return
    
    # Добавляем водороды (обязательно для правильной геометрии)
    mol = Chem.AddHs(mol)
    
    # Генерируем 3D координаты (используем современный метод ETKDGv3)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    
    # Оптимизируем структуру силовым полем UFF
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        print("UFF Optimization failed, saving embedded structure anyway.")
    
    # Сохраняем в SDF
    writer = Chem.SDWriter(output_file)
    writer.write(mol)
    writer.close()
    print(f"✅ 3D structure saved to {output_file}")

if __name__ == "__main__":
    # Берем твоего лучшего кандидата №76
    smi = "C[NH1]CNCPOCl" 
    make_3d(smi, "candidate.sdf")
