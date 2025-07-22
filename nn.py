import torch
import torch.nn as nn

from dataset_operations import stoi

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (seq_len, batch)
        embed = self.embedding(x)             # (seq_len, batch, hidden)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out)                    # (seq_len, batch, vocab_size)
        return out, hidden


model = CharLSTM(vocab_size=len(stoi), hidden_size=128)