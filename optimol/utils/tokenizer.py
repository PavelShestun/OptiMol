import torch
from torch.utils.data import Dataset
import selfies as sf
from loguru import logger

class SelfiesTokenizer:
    def __init__(self, smiles_list=None, vocab=None):
        if vocab:
            self.vocab = vocab
        else:
            logger.info("Building SELFIES vocabulary...")
            all_tokens = set()
            for s in smiles_list:
                try:
                    res = sf.encoder(s)
                    all_tokens.update(sf.get_alphabet_from_selfies([res]))
                except: continue
            
            self.vocab = {token: i for i, token in enumerate(sorted(list(all_tokens)))}
            self.vocab['[PAD]'] = len(self.vocab)
            self.vocab['[START]'] = len(self.vocab)
            self.vocab['[END]'] = len(self.vocab)
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, smiles):
        try:
            selfie = sf.encoder(smiles)
            tokens = sf.get_alphabet_from_selfies([selfie])
            return [self.vocab['[START]']] + [self.vocab[t] for t in tokens if t in self.vocab] + [self.vocab['[END]']]
        except: return None

    def decode(self, ids):
        res = ""
        for i in ids:
            if torch.is_tensor(i): i = i.item()
            token = self.inverse_vocab.get(i, '[PAD]')
            if token in ['[START]', '[END]', '[PAD]']: continue
            res += token
        return res

class SelfiesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer):
        self.data = []
        for s in smiles_list:
            encoded = tokenizer.encode(s)
            if encoded:
                self.data.append(torch.tensor(encoded, dtype=torch.long))
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item[:-1], item[1:]

def collate_fn(batch, pad_value):
    inputs, targets = zip(*batch)
    inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_value)
    targets_pad = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_value)
    return inputs_pad, targets_pad
