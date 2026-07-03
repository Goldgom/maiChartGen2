import torch
import torch.nn as nn

class AttenMask(nn.Module):
    def __init__(self, mask_type='causal'):
        super().__init__()
        self.mask_type = mask_type

    def forward(self, seq_len):
        if self.mask_type == 'causal':
            # 生成一个下三角矩阵，表示未来位置不可见
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        return mask  # [seq_len, seq_len]，True 表示可见，False 表示不可见
