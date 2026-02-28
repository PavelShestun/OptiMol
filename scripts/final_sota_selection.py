import torch
import pandas as pd
import selfies as sf
from optimol.models.generator_rnn import SELFIESGenerator
from optimol.models.predictor_gnn import Keap1PredictorGNN
from optimol.utils.graph_utils import create_pytorch_geom_dataset
from torch_geometric.loader import DataLoader as GNNDataLoader
from optimol.utils.chemistry import get_full_properties
from tqdm import tqdm
import os

def select_sota_candidates(n_samples=2000): # Увеличим выборку до 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    
    checkpoint = torch.load("models/checkpoints/generator_pretrain.pt", map_location=device)
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inverse_vocab']
    
    gen = SELFIESGenerator(vocab_size=len(vocab)).to(device)
    gen.load_state_dict(torch.load("models/checkpoints/generator_rl_final.pt", map_location=device))
    gen.eval()
    
    pred = Keap1PredictorGNN(node_features=6).to(device)
    pred.load_state_dict(torch.load("models/checkpoints/predictor_gnn_final.pt", map_location=device))
    pred.eval()
    
    candidates = []
    print(f"SOTA Phase: Sampling {n_samples} candidates...")
    
    for _ in tqdm(range(n_samples)):
        state = torch.tensor([[vocab['[START]']]]).to(device)
        hidden = None
        tokens = []
        for _ in range(100):
            logits, hidden = gen(state, hidden)
            probs = torch.softmax(logits[:, -1, :] / 1.2, dim=-1) # Больше креативности
            action = torch.multinomial(probs, num_samples=1).item()
            if action == vocab['[END]']: break
            tokens.append(inv_vocab[action])
            state = torch.tensor([[action]]).to(device)
        
        try:
            smiles = sf.decoder("".join(tokens))
            props = get_full_properties(smiles)
            # Снижаем планку до 8 атомов, чтобы увидеть лучшие структуры
            if props and props['num_atoms'] >= 8: 
                temp_df = pd.DataFrame([{'canonical_smiles': smiles, 'pIC50': 0}])
                pyg_data = create_pytorch_geom_dataset(temp_df)
                # Защита от пустых графов
                batch = next(iter(GNNDataLoader(pyg_data, batch_size=1))).to(device)
                if batch.edge_index.shape[1] > 0:
                    with torch.no_grad():
                        props['pIC50'] = pred(batch).item()
                    candidates.append(props)
        except: continue
    
    if not candidates:
        print("No candidates found with num_atoms >= 8. Try lower filter.")
        return

    df = pd.DataFrame(candidates).drop_duplicates(subset=['smiles'])
    
    # Сортируем по pIC50 (активность) и QED (лекарственность)
    df = df.sort_values(by=['pIC50', 'qed'], ascending=False)
    df.to_csv("results/final_sota_candidates.csv", index=False)
    
    print(f"\nSuccessfully found {len(df)} candidates.")
    print("\n--- TOP 10 SOTA CANDIDATES ---")
    print(df[['smiles', 'pIC50', 'qed', 'num_atoms']].head(10))

if __name__ == "__main__":
    select_sota_candidates(2000)
