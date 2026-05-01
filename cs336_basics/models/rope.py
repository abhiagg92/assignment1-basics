import torch
from torch import nn
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        angles = torch.tensor([[i/(theta**((2*k-2)/d_k)) for k in range(1, d_k//2+1)] for i in range(max_seq_len)], device=device)
        cos_angles = torch.cos(angles).repeat_interleave(2, dim=1)
        sin_angles = torch.sin(angles).repeat_interleave(2, dim=1)
        sin_angles[:, ::2] *= -1
        self.register_buffer('cos_angles', cos_angles, persistent=False)
        self.register_buffer('sin_angles', sin_angles, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos_angles = self.cos_angles[token_positions]
        sin_angles = self.sin_angles[token_positions]

        x_swapped = rearrange(x, '... (d1 d2) -> ... d1 d2', d2=2).flip(-1)
        x_swapped = rearrange(x_swapped, '... d1 d2 -> ... (d1 d2)')
        x_rope = x*cos_angles+x_swapped*sin_angles
        return x_rope