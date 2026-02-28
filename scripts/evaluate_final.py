import torch
import pandas as pd
import selfies as sf
from optimol.models.generator_rnn import SELFIESGenerator
from optimol.utils.chemistry import get_full_properties
from loguru import logger
import os

def run_evaluation():
    os.makedirs("results", exist_ok=True)
    
    # Загрузка словаря и модели
    checkpoint = torch.load("models/checkpoints/generator_pretrain.pt")
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inverse_vocab']
    
    gen = SELFIESGenerator(vocab_size=len(vocab))
    try:
        gen.load_state_dict(torch.load("models/checkpoints/generator_rl_final.pt"))
        logger.info("Loaded RL-tuned model weights.")
    except:
        logger.warning("RL weights not found, using pre-trained weights.")
    
    gen.eval()
    candidates = []
    logger.info("Generating 100 final candidates...")
    
    for _ in range(100):
        state = torch.tensor([[vocab['[START]']]])
        hidden = None
        tokens = []
        for _ in range(100):
            logits, hidden = gen(state, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
            if action == vocab['[END]']: break
            tokens.append(inv_vocab[action])
            state = torch.tensor([[action]])
        
        try:
            selfies_str = "".join(tokens)
            smiles = sf.decoder(selfies_str)
            if smiles:
                props = get_full_properties(smiles)
                if props:
                    candidates.append(props)
        except:
            continue
    
    if not candidates:
        logger.error("No valid molecules generated! Try increasing generation count or check reward function.")
        return

    df = pd.DataFrame(candidates).drop_duplicates(subset=['smiles'])
    
    # Проверяем наличие колонок перед сортировкой
    cols_to_sort = [c for c in ['qed', 'cns_mpo'] if c in df.columns]
    if cols_to_sort:
        df = df.sort_values(by=cols_to_sort, ascending=False)
    
    df.to_csv("results/sota_candidates.csv", index=False)
    logger.success(f"Found {len(df)} unique candidates.")
    
    print("\n--- TOP 10 GENERATED MOLECULES ---")
    # Выводим SMILES и основные метрики
    print(df[['smiles', 'qed', 'num_atoms']].head(10))

if __name__ == "__main__":
    run_evaluation()
