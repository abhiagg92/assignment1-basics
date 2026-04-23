import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.square().sum(-1, keepdim=True)/self.d_model+self.eps)
        x = (x/rms)*self.gain
        return x.to(in_dtype)