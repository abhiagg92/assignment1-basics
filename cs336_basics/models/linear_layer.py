import torch
from torch import nn
from einops import einsum



class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        init_std = (2/(in_features+out_features))**0.5

        W = torch.empty((in_features, out_features), dtype=dtype, device=device)
        nn.init.trunc_normal_(W, std=init_std, a=-3*init_std, b=3*init_std)
        self.W = nn.Parameter(W)
    
    def forward(self, x: torch.Tensor):
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")