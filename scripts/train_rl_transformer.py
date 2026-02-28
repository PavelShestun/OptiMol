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

from optimol.models.generator_transformer import MolTransformer
from optimol.models.predictor_gnn import Keap1PredictorGNN
from optimol.utils.graph_utils import create_pytorch_geom_dataset
from torch_geometric.loader import DataLoader as GNNDataLoader
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from optimol.utils.sota_metrics import calculate_sa_score, get_diversity_penalty

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_history = deque(maxlen=500) # Для контроля разнообразия

def calculate_sota_reward(smiles, predictor, device):
    if not smiles or smiles == "": return 0.01 # Минимальный бонус за непустую строку
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 0.0
        
        n_atoms = mol.GetNumAtoms()
        
        # 1. Базовая награда просто за то, что молекула ПРАВИЛЬНАЯ
        # Это поможет модели уйти от нулевых градиентов
        validity_reward = 0.5 
        
        # 2. Активность (pIC50)
        temp_df = pd.DataFrame([{'canonical_smiles': smiles, 'pIC50': 0}])
        pyg_data = create_pytorch_geom_dataset(temp_df)
        loader = GNNDataLoader(pyg_data, batch_size=1)
        predictor.eval()
        with torch.no_grad():
            batch = next(iter(loader)).to(device)
            pic50 = predictor(batch).item() if batch.edge_index.shape[1] > 0 else 3.5
        
        # 3. QED и SA
        qed_val = QED.qed(mol)
        sa_score = calculate_sa_score(mol)
        sa_reward = (10 - sa_score) / 10
        
        # 4. Прогрессивная награда за размер (Эскалатор)
        # Мы НЕ обнуляем за маленькие молекулы, а просто даем меньше
        size_reward = n_atoms * 0.2
        
        # Сборная награда
        base_score = (pic50 / 10.0) * qed_val * sa_reward
        total_reward = validity_reward + base_score + size_reward
        
        # Бонусы за кольца
        if mol.GetRingInfo().NumRings() >= 1: total_reward += 2.0
        
        return total_reward
    except:
        return 0.0

def train_rl():
    os.makedirs("models/checkpoints", exist_ok=True)
    wandb.init(project="optimol-sota", name="transformer-kan-rl")
    
    # Загрузка Трансформера
    checkpoint = torch.load("models/checkpoints/transformer_pretrain.pt", map_location=device)
    vocab = checkpoint['vocab']
    inv_vocab = {v: k for k, v in vocab.items()}
    
    gen = MolTransformer(vocab_size=len(vocab)).to(device)
    gen.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(gen.parameters(), lr=0.00003)
    
    # Загрузка Предиктора
    pred = Keap1PredictorGNN(node_features=6).to(device)
    pred.load_state_dict(torch.load("models/checkpoints/predictor_gnn_final.pt", map_location=device))
    pred.eval()
    
    logger.info("Starting SOTA RL with Transformer and GNN-KAN...")
    
    for epoch in range(100):
        gen.train()
        batch_rewards, batch_log_probs = [], []
        
        for _ in range(32): # Batch size
            sequence = torch.tensor([[vocab['[START]']]]).to(device)
            log_probs = []
            tokens = []
            
            for _ in range(80): # Max length
                logits = gen(sequence)
                probs = torch.softmax(logits[:, -1, :] / 1.1, dim=-1) # Temp sampling
                
                action = torch.multinomial(probs, num_samples=1)
                log_probs.append(torch.log(probs[0, action]))
                
                token = inv_vocab[action.item()]
                if token == '[END]': break
                
                tokens.append(token)
                sequence = torch.cat([sequence, action], dim=1)
            
            smiles = ""
            try:
                smiles = sf.decoder("".join(tokens))
            except: pass
            
            reward = calculate_sota_reward(smiles, pred, device)
            batch_rewards.append(reward)
            batch_log_probs.append(log_probs)
            
            if reward > 8.0:
                logger.success(f"💎 SOTA Molecule: {smiles} | Reward: {reward:.2f}")

        # Policy Gradient Update
        optimizer.zero_grad()
        baseline = np.mean(batch_rewards)
        
        if np.std(batch_rewards) < 1e-6:
            rewards_to_use = [r + np.random.uniform(0, 0.01) for r in batch_rewards]
            baseline = np.mean(rewards_to_use)
        else:
            rewards_to_use = batch_rewards

        loss_list = []
        for i in range(len(batch_rewards)):
            advantage = batch_rewards[i] - baseline
            for lp in batch_log_probs[i]:
                loss_list.append(-lp * advantage)
        
        if loss_list:
            loss = torch.cat(loss_list).mean()
            loss.backward()
            optimizer.step()
            
        logger.info(f"Epoch {epoch} | Avg Reward: {baseline:.4f} | Loss: {loss.item():.4f}")
        wandb.log({"avg_reward": baseline, "loss": loss.item()})
        
        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), "models/checkpoints/transformer_rl_final.pt")

if __name__ == "__main__":
    train_rl()
