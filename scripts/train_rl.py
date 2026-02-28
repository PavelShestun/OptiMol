import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import selfies as sf
from tqdm import tqdm
from loguru import logger
import wandb
import os
from collections import deque

from optimol.utils.sota_metrics import calculate_sa_score, get_diversity_penalty
from optimol.models.generator_transformer import MolTransformer
from optimol.models.predictor_gnn import Keap1PredictorGNN
from optimol.utils.graph_utils import create_pytorch_geom_dataset
from torch_geometric.loader import DataLoader as GNNDataLoader
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_buffer = deque(maxlen=200)
global_history = deque(maxlen=1000)

def calculate_reward(smiles, predictor, device):
    if not smiles or smiles == "": return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 0.0
        
        n_atoms = mol.GetNumAtoms()
        n_rings = mol.GetRingInfo().NumRings()
        
        # 1. Предиктор pIC50
        temp_df = pd.DataFrame([{'canonical_smiles': smiles, 'pIC50': 0}])
        pyg_data = create_pytorch_geom_dataset(temp_df)
        loader = GNNDataLoader(pyg_data, batch_size=1)
        predictor.eval()
        with torch.no_grad():
            batch = next(iter(loader)).to(device)
            if batch.edge_index.shape[1] == 0: 
                pic50 = 3.0
            else: 
                pic50 = predictor(batch).item()
        
        qed_score = QED.qed(mol)
        
        # --- ПЛАВНАЯ СИСТЕМА НАГРАД (ESCAlATOR) ---
        reward = (pic50 * 0.2) + (qed_score * 1.0)
        
        # Поощряем рост
        reward += (n_atoms * 0.15) 
        
        # Поощряем кольца
        if n_rings > 0:
            reward += 3.0
            if n_rings > 2: reward += 2.0
            
        # Бонус за адекватный вес
        mw = Descriptors.MolWt(mol)
        if 250 < mw < 550:
            reward += 5.0

        return max(reward, 0.0)
    except:
        return 0.0

def calculate_sota_reward(smiles, predictor, device):
    if not smiles or smiles == "": return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 0.0
        
        # 1. Предиктор pIC50 (в идеале - ансамбль)
        pic50 = 4.0 # База
        temp_df = pd.DataFrame([{'canonical_smiles': smiles, 'pIC50': 0}])
        pyg_data = create_pytorch_geom_dataset(temp_df)
        batch = next(iter(GNNDataLoader(pyg_data, batch_size=1))).to(device)
        if batch.edge_index.shape[1] > 0:
            pic50 = predictor(batch).item()

        # 2. Метрики качества
        qed = QED.qed(mol)
        sa_score = calculate_sa_score(mol) # 1 (легко) - 10 (невозможно)
        sa_reward = (10 - sa_score) / 10 # Инвертируем: 1 - отлично, 0 - плохо
        
        # 3. Штраф за повторы (Diversity)
        div_penalty = get_diversity_penalty(smiles, list(global_history))
        global_history.append(smiles)

        # 4. Итоговая SOTA формула
        # Мы перемножаем, а не складываем! Если один параметр 0, всё 0.
        reward = (pic50 / 10.0) * qed * sa_reward * div_penalty
        
        # Бонус за "взрослый" размер и кольца (обязательно для KEAP1)
        if mol.GetRingInfo().NumRings() >= 2: reward += 2.0
        if 20 < mol.GetNumAtoms() < 45: reward += 3.0
        
        return max(reward, 0.0)
    except: return 0.0


def train_rl():
    os.makedirs("models/checkpoints", exist_ok=True)
    wandb.init(project="optimol-rl", name="rl-curriculum-fixed")
    
    # Загрузка
    checkpoint = torch.load("models/checkpoints/generator_pretrain.pt", map_location=device)
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inverse_vocab']
    
    gen = SELFIESGenerator(vocab_size=len(vocab)).to(device)
    gen.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001)
    
    pred = Keap1PredictorGNN(node_features=6).to(device)
    pred_path = "models/checkpoints/predictor_gnn_final.pt"
    if os.path.exists(pred_path):
        pred.load_state_dict(torch.load(pred_path, map_location=device))
        logger.success("Predictor loaded.")
    else:
        logger.warning("Predictor weights not found! Training will be suboptimal.")
    pred.eval()
    
    logger.info("Starting RL Phase: Escalator (Fixed)...")
    
    for epoch in range(50):
        gen.train()
        batch_rewards = []
        batch_log_probs = []
        
        for _ in range(48):
            state = torch.tensor([[vocab['[START]']]]).to(device)
            hidden = None
            log_probs, tokens = [], []
            
            for _ in range(80):
                logits, hidden = gen(state, hidden)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                action = torch.multinomial(probs, num_samples=1)
                log_probs.append(torch.log(probs[0, action]))
                token = inv_vocab[action.item()]
                if token == '[END]': break
                tokens.append(token)
                state = action
            
            smiles = ""
            try:
                smiles = sf.decoder("".join(tokens))
            except: pass
            
            reward = calculate_reward(smiles, pred, device)
            if reward > 2.0: 
                replay_buffer.append(smiles)
                if reward > 10.0:
                    logger.success(f"High Reward! {reward:.2f} | {smiles}")
            
            batch_rewards.append(reward)
            batch_log_probs.append(log_probs)
            
        optimizer.zero_grad()
        avg_reward = np.mean(batch_rewards) # РАССЧИТЫВАЕМ ТУТ
        
        loss_list = []
        for i in range(len(batch_rewards)):
            advantage = batch_rewards[i] - avg_reward
            for lp in batch_log_probs[i]:
                loss_list.append(-lp * advantage)
        
        if loss_list:
            loss = torch.cat(loss_list).mean()
            loss.backward()
            optimizer.step()
        
        # Логирование без walrus-оператора
        logger.info(f"Epoch {epoch:02d} | Avg Reward: {avg_reward:.4f} | Buffer: {len(replay_buffer)}")
        wandb.log({"avg_reward": avg_reward, "buffer": len(replay_buffer)})

    torch.save(gen.state_dict(), "models/checkpoints/generator_rl_final.pt")
    logger.success("RL complete.")

if __name__ == "__main__":
    train_rl()
