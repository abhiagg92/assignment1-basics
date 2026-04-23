import torch
from torch import nn, Tensor
from einops import einsum
from jaxtyping import Float

from cs336_basics.models import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.l1 = Linear(d_model, d_ff, device, dtype)
        self.l2 = Linear(d_ff, d_model, device, dtype)
        self.l3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: Float[Tensor, "batch seq_len d_model"]):
        x1 = self.l1(x)
        x1 = x1*torch.sigmoid(x1)
        x3 = self.l3(x)
        return self.l2(x1*x3)