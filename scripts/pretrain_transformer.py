import torch
import pandas as pd
from torch.utils.data import DataLoader
from optimol.models.generator_transformer import MolTransformer
from optimol.utils.tokenizer import SelfiesTokenizer, SelfiesDataset, collate_fn
from functools import partial
from tqdm import tqdm
from loguru import logger

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("data/raw/pretrain_data.csv")
    
    # Используем вокабуляр из прошлого шага для консистентности
    old_checkpoint = torch.load("models/checkpoints/generator_pretrain.pt", map_location='cpu')
    vocab = old_checkpoint['vocab']
    tokenizer = SelfiesTokenizer(vocab=vocab)
    
    dataset = SelfiesDataset(df['canonical_smiles'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, 
                        collate_fn=partial(collate_fn, pad_value=vocab['[PAD]']))
    
    model = MolTransformer(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) # AdamW лучше для Transformer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
    
    logger.info("Starting Transformer Pre-training...")
    for epoch in range(20):
        model.train()
        pbar = tqdm(loader)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
    torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab}, 
               "models/checkpoints/transformer_pretrain.pt")

if __name__ == "__main__":
    train()
