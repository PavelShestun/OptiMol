import torch
import selfies as sf
from optimol.models.generator_rnn import SELFIESGenerator
from optimol.utils.tokenizer import SelfiesTokenizer

def sample(num_samples=10):
    checkpoint = torch.load("models/checkpoints/generator_pretrain.pt")
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inverse_vocab']
    
    model = SELFIESGenerator(vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    generated = []
    for _ in range(num_samples):
        input_seq = torch.tensor([[vocab['[START]']]])
        hidden = None
        molecule_tokens = []
        
        for _ in range(100): # Макс длина 100 символов
            logits, hidden = model(input_seq, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == vocab['[END]']:
                break
            
            molecule_tokens.append(inv_vocab[next_token])
            input_seq = torch.tensor([[next_token]])
            
        selfies_str = "".join(molecule_tokens)
        try:
            smiles = sf.decoder(selfies_str)
            generated.append(smiles)
        except:
            continue
            
    return generated

if __name__ == "__main__":
    mols = sample(5)
    print("Generated Molecules:")
    for m in mols:
        print(m)
