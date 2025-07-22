import torch
import torch.nn as nn

from dataset_operations import stoi


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # I have it as batch_first=True because the dataloader returns (batch, seq_len)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        embed = self.embedding(x)  # (batch, seq_len, hidden)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out)  # (batch, seq_len, vocab_size)
        return out, hidden
