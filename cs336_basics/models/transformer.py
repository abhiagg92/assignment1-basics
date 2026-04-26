import torch
from torch import Tensor, nn
from jaxtyping import Float

from cs336_basics.models import RMSNorm, MultiHeadAttention, SwiGLU


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()

        self.device = device
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]):
        batch, seq_len, _ = x.shape
        norm1 = self.ln1(x)
        token_positions = torch.arange(seq_len, dtype=torch.int, device=self.device)  # shape (seq_len,)
        token_positions = token_positions.unsqueeze(0).repeat(batch, 1)  # shape (batch, seq_len)
        x += self.attn(norm1, token_positions)

        norm2 = self.ln2(x)
        x += self.ffn(norm2)
        return x