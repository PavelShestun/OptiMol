from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import os

def draw_top_molecules():
    df = pd.read_csv("results/SOTA_MINER_REPORT.csv")
    top_5 = df.head(5)
    
    mols = []
    legends = []
    
    for _, row in top_5.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            mols.append(mol)
            legends.append(f"pIC50: {row['pIC50_pred']:.2f}, QED: {row['qed']:.2f}")
    
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legends)
    img.save("results/TOP_SOTA_CANDIDATES.png")
    print("🎨 Image saved to results/TOP_SOTA_CANDIDATES.png")

if __name__ == "__main__":
    draw_top_molecules()
