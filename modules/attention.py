import math

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, Q=512, K=512, V=512, dmodel=512, precision=torch.float32, atten_mask=None):
        super().__init__()
        self.Q = nn.Linear(Q, Q, dtype=precision)
        self.K = nn.Linear(K, K, dtype=precision)
        self.V = nn.Linear(V, V, dtype=precision)
        self.atten_mask = atten_mask

    def forward(self, Q, K, V):
        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        if self.atten_mask is not None:
            mask = self.atten_mask(scores.size(-1)).to(scores.device)
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output
