import torch
import torch.nn as nn

class SELFIESGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, num_layers=3):
        super(SELFIESGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
