import torch
import pandas as pd
import selfies as sf
from optimol.models.generator_rnn import SELFIESGenerator
from optimol.utils.chemistry import get_full_properties
from loguru import logger
from tqdm import tqdm
import os

def sota_generation(num_to_sample=5000):
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load("models/checkpoints/generator_pretrain.pt", map_location=device)
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inverse_vocab']
    
    gen = SELFIESGenerator(vocab_size=len(vocab)).to(device)
    gen.load_state_dict(torch.load("models/checkpoints/generator_rl_final.pt", map_location=device))
    gen.eval()
    
    results = []
    logger.info(f"SOTA Phase: Sampling {num_to_sample} molecules to find outliers...")

    for _ in tqdm(range(num_to_sample)):
        state = torch.tensor([[vocab['[START]']]]).to(device)
        hidden = None
        tokens = []
        
        # Увеличиваем температуру до 1.2 для большей креативности
        for _ in range(120): 
            logits, hidden = gen(state, hidden)
            probs = torch.softmax(logits[:, -1, :] / 1.2, dim=-1) 
            action = torch.multinomial(probs, num_samples=1).item()
            if action == vocab['[END]']: break
            tokens.append(inv_vocab[action])
            state = torch.tensor([[action]]).to(device)
        
        try:
            smiles = sf.decoder("".join(tokens))
            props = get_full_properties(smiles)
            # ЖЕСТКИЙ ФИЛЬТР: Только молекулы больше 15 атомов
            if props and props['num_atoms'] >= 12:
                results.append(props)
        except: continue

    if not results:
        logger.warning("No large molecules found. Lowering filter to 10 atoms...")
        # (повторить поиск с меньшим фильтром если пусто)

    df = pd.DataFrame(results).drop_duplicates(subset=['smiles'])
    # Сортируем по QED и весу одновременно
    df['sota_score'] = df['qed'] * (df['num_atoms'] / 10) 
    df = df.sort_values(by='sota_score', ascending=False)
    
    df.to_csv("results/sota_final_selection.csv", index=False)
    logger.success(f"Generated {len(df)} SOTA candidates!")
    print(df[['smiles', 'qed', 'num_atoms', 'mw']].head(10))

if __name__ == "__main__":
    sota_generation(5000)
