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

def run_sota_generation(n_samples=500):
    device = torch.device("cpu") # Используем CPU для стабильности финальной обработки
    os.makedirs("results", exist_ok=True)
    
    logger.info("🚀 Initializing SOTA Report Generation...")

    # 1. Загрузка вокабуляра и модели
    checkpoint_path = "models/checkpoints/transformer_pretrain.pt"
    rl_weights_path = "models/checkpoints/transformer_rl_final.pt"
    
    if not os.path.exists(rl_weights_path):
        logger.error("RL weights not found! Run scripts/train_rl_transformer.py first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']
    inv_vocab = {v: k for k, v in vocab.items()}
    
    gen = MolTransformer(vocab_size=len(vocab)).to(device)
    gen.load_state_dict(torch.load(rl_weights_path, map_location=device))
    gen.eval()
    
    # 2. Загрузка Предиктора GNN-KAN
    pred = Keap1PredictorGNN(node_features=6).to(device)
    pred.load_state_dict(torch.load("models/checkpoints/predictor_gnn_final.pt", map_location=device))
    pred.eval()
    
    candidates = []

    # 3. Генерация с "консервативной" температурой (0.8) для избежания хэкинга
    logger.info(f"Generating {n_samples} candidates with optimized sampling...")
    for _ in tqdm(range(n_samples)):
        sequence = torch.tensor([[vocab['[START]']]]).to(device)
        tokens = []
        for _ in range(100):
            with torch.no_grad():
                logits = gen(sequence)
            
            # Температура 0.8 делает молекулы более "правильными" и менее хаотичными
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            
            token = inv_vocab[action.item()]
            if token == '[END]': break
            tokens.append(token)
            sequence = torch.cat([sequence, action], dim=1)
        
        try:
            smiles = sf.decoder("".join(tokens))
            props = get_full_properties(smiles)
            
            if props and 15 < props['num_atoms'] < 60:
                # Расчет pIC50 через GNN-KAN
                temp_df = pd.DataFrame([{'canonical_smiles': smiles, 'pIC50': 0}])
                pyg_data = create_pytorch_geom_dataset(temp_df)
                batch = next(iter(GNNDataLoader(pyg_data, batch_size=1))).to(device)
                
                with torch.no_grad():
                    props['pIC50_pred'] = pred(batch).item()
                
                # SOTA метрики
                props['sa_score'] = calculate_sa_score(torch.utils.data.dataset.Chem.MolFromSmiles(smiles))
                props['cns_mpo'] = calculate_cns_mpo(smiles)
                # Ligand Efficiency (LE)
                props['LE'] = (1.37 * props['pIC50_pred']) / props['num_atoms']
                
                candidates.append(props)
        except: continue

    # 4. Сборка финальной таблицы
    if not candidates:
        logger.error("No valid candidates found! Try retraining RL for more epochs.")
        return

    df = pd.DataFrame(candidates).drop_duplicates(subset=['smiles'])
    
    # Фильтрация по "аптечным" стандартам (Lipinski-like)
    df = df[
        (df['mw'] < 600) & 
        (df['logp'] < 5) & 
        (df['qed'] > 0.3)
    ]
    
    # Сортировка по pIC50 и Лигандной эффективности
    df = df.sort_values(by=['pIC50_pred', 'LE'], ascending=False)
    
    output_path = "results/SOTA_FINAL_REPORT.csv"
    df.to_csv(output_path, index=False)
    
    logger.success(f"SOTA Report generated! Saved {len(df)} molecules to {output_path}")
    
    print("\n" + "="*50)
    print("      TOP 5 SOTA CANDIDATES FOR KEAP1")
    print("="*50)
    print(df[['smiles', 'pIC50_pred', 'LE', 'qed', 'cns_mpo']].head(5))
    print("="*50)

if __name__ == "__main__":
    run_sota_generation(500)
