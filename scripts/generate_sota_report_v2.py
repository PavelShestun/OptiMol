import torch
import pandas as pd
import selfies as sf
import numpy as np
from tqdm import tqdm
from loguru import logger
import os

from optimol.models.generator_transformer import MolTransformer
from optimol.models.predictor_gnn import Keap1PredictorGNN
from optimol.utils.graph_utils import create_pytorch_geom_dataset
from torch_geometric.loader import DataLoader as GNNDataLoader
from optimol.utils.chemistry import get_full_properties, calculate_cns_mpo
from optimol.utils.sota_metrics import calculate_sa_score
from rdkit import Chem

def run_sota_mining(n_samples=1000):
    device = torch.device("cpu")
    os.makedirs("results", exist_ok=True)
    
    logger.info("🔍 SOTA Mining: Searching for the best fragments and leads...")

    # Загрузка моделей
    checkpoint = torch.load("models/checkpoints/transformer_pretrain.pt", map_location=device)
    vocab = checkpoint['vocab']
    inv_vocab = {v: k for k, v in vocab.items()}
    
    gen = MolTransformer(vocab_size=len(vocab)).to(device)
    gen.load_state_dict(torch.load("models/checkpoints/transformer_rl_final.pt", map_location=device))
    gen.eval()
    
    pred = Keap1PredictorGNN(node_features=6).to(device)
    pred.load_state_dict(torch.load("models/checkpoints/predictor_gnn_final.pt", map_location=device))
    pred.eval()
    
    candidates = []

    # Увеличим температуру до 1.0, чтобы выйти из зоны "зацикливания" на длинных цепях
    for _ in tqdm(range(n_samples)):
        sequence = torch.tensor([[vocab['[START]']]]).to(device)
        tokens = []
        for _ in range(80): # Ограничиваем длину принудительно до 80 токенов
            with torch.no_grad():
                logits = gen(sequence)
            probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            token = inv_vocab[action.item()]
            if token == '[END]': break
            tokens.append(token)
            sequence = torch.cat([sequence, action], dim=1)
        
        try:
            smiles = sf.decoder("".join(tokens))
            mol = Chem.MolFromSmiles(smiles)
            if not mol: continue
            
            # СНИЖАЕМ ПОРОГ: ищем молекулы от 10 атомов
            n_atoms = mol.GetNumAtoms()
            if n_atoms >= 10:
                props = get_full_properties(smiles)
                
                # Оценка pIC50
                temp_df = pd.DataFrame([{'canonical_smiles': smiles, 'pIC50': 0}])
                pyg_data = create_pytorch_geom_dataset(temp_df)
                batch = next(iter(GNNDataLoader(pyg_data, batch_size=1))).to(device)
                with torch.no_grad():
                    props['pIC50_pred'] = pred(batch).item()
                
                props['sa_score'] = calculate_sa_score(mol)
                props['LE'] = (1.37 * props['pIC50_pred']) / n_atoms
                candidates.append(props)
        except: continue

    if not candidates:
        logger.error("Still no candidates. This means RL diverged. Check models/checkpoints/!")
        return

    df = pd.DataFrame(candidates).drop_duplicates(subset=['smiles'])
    
    # Сортируем по pIC50_pred и QED (без жесткой фильтрации)
    df = df.sort_values(by=['pIC50_pred', 'qed'], ascending=False)
    
    output_path = "results/SOTA_MINER_REPORT.csv"
    df.to_csv(output_path, index=False)
    
    logger.success(f"Successfully mined {len(df)} candidates!")
    print("\n--- TOP 10 MINED CANDIDATES ---")
    print(df[['smiles', 'pIC50_pred', 'qed', 'num_atoms', 'mw']].head(10))

if __name__ == "__main__":
    run_sota_mining(1000)
