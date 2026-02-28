import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from loguru import logger
import os

from optimol.models.generator_rnn import SELFIESGenerator
from optimol.utils.tokenizer import SelfiesTokenizer

class SelfiesDataset(Dataset):
    def __init__(self, selfies_list, tokenizer):
        self.data = [torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in selfies_list]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Для обучения предсказанию следующего токена:
        # Input: [START, A, B, C]
        # Target: [A, B, C, END]
        item = self.data[idx]
        return item[:-1], item[1:]

def collate_fn(batch):
    # Добавляем padding, чтобы все строки в батче были одной длины
    inputs, targets = zip(*batch)
    inputs_pad = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.vocab['[PAD]'])
    targets_pad = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.vocab['[PAD]'])
    return inputs_pad, targets_pad

def train_generator():
    os.makedirs("models/checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Загрузка данных
    df = pd.read_csv("data/raw/pretrain_data.csv")
    global tokenizer # Чтобы collate_fn видел его
    tokenizer = SelfiesTokenizer(df['canonical_smiles'].tolist())
    
    dataset = SelfiesDataset(tokenizer.selfies_list, tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    # 2. Инициализация модели
    model = SELFIESGenerator(vocab_size=len(tokenizer.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['[PAD]'])
    
    # 3. Цикл обучения
    epochs = 10 
    logger.info(f"Starting pre-training on {len(dataset)} molecules...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            
            # Поворачиваем размерности для CrossEntropy: (N, C, L)
            loss = criterion(logits.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
        
        # Сохранение весов и токенизатора
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': tokenizer.vocab,
            'inverse_vocab': tokenizer.inverse_vocab
        }, "models/checkpoints/generator_pretrain.pt")

if __name__ == "__main__":
    train_generator()
