import torch
from torch import nn, Tensor
from einops import einsum
from jaxtyping import Float

from cs336_basics.models import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: Float[Tensor, "batch seq_len d_model"]):
        x1 = self.w1(x)
        x1 = x1*torch.sigmoid(x1)
        x3 = self.w3(x)
        return self.w2(x1*x3)