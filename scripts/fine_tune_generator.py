import torch
import pandas as pd
from torch.utils.data import DataLoader
from optimol.models.generator_rnn import SELFIESGenerator
from optimol.utils.tokenizer import SelfiesTokenizer, SelfiesDataset, collate_fn
from functools import partial
import os

def fine_tune():
    os.makedirs("models/checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Загрузка данных KEAP1
    df = pd.read_csv("data/processed/keap1_cleaned.csv")
    
    # 2. Загрузка пре-трейна для получения вокабуляра
    checkpoint = torch.load("models/checkpoints/generator_pretrain.pt", map_location=device)
    vocab = checkpoint['vocab']
    
    tokenizer = SelfiesTokenizer(vocab=vocab)
    
    # Создаем датасет
    dataset = SelfiesDataset(df['canonical_smiles'].tolist(), tokenizer)
    
    # Настраиваем collate_fn с правильным padding
    custom_collate = partial(collate_fn, pad_value=vocab['[PAD]'])
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    
    # 3. Инициализация модели
    model = SELFIESGenerator(vocab_size=len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Замораживаем веса эмбеддингов (опционально, для стабильности на малых данных)
    # model.embedding.weight.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
    
    print(f"Starting Fine-tuning on {len(dataset)} KEAP1 molecules...")
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "models/checkpoints/generator_finetuned_keap1.pt")
    print("Fine-tuning complete! Saved to models/checkpoints/generator_finetuned_keap1.pt")

if __name__ == "__main__":
    fine_tune()

