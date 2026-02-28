import torch
from rdkit import Chem
from torch_geometric.data import Data

def atom_features(atom):
    # Базовый набор признаков атома для GNN
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization().numerator,
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
    ]

def create_pytorch_geom_dataset(df):
    data_list = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if not mol: continue
        
        # Узлы (атомы)
        nodes = [atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(nodes, dtype=torch.float)
        
        # Ребра (связи)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append((i, j))
            edges.append((j, i))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Целевое значение pIC50
        y = torch.tensor([row['pIC50']], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list
