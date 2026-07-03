import torch
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, dmodel=512, precision=torch.float32, base=10000.0):
        super().__init__()
        if dmodel % 2 != 0:
            raise ValueError("RoPE requires an even feature dimension.")

        self.dmodel = dmodel
        self.precision = precision
        self.base = float(base)

    def _build_angles(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        dim_ids = torch.arange(0, self.dmodel, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.base ** (dim_ids / self.dmodel))
        return torch.outer(positions, inv_freq)

    def forward(self, x):
        # x: [batch_size, heads, seq_len, head_dim]
        if x.size(-1) != self.dmodel:
            raise ValueError("Input feature dimension must match Rope.dmodel.")

        seq_len = x.size(-2)
        angles = self._build_angles(seq_len, x.device, x.dtype if x.is_floating_point() else self.precision)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        return rotated
