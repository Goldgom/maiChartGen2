import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, dmodel=512, dff=2048, precision=torch.float32, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dmodel, dff, dtype=precision)
        self.linear2 = nn.Linear(dff, dmodel, dtype=precision)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
