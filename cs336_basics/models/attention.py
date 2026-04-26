import torch
from torch import Tensor, nn
from jaxtyping import Float, Bool
from einops import einsum, rearrange

from cs336_basics.models import softmax, Linear, RotaryPositionalEmbedding


def scaled_dot_product_attention(
        Q: Float[Tensor, "batch_size ... seq_len d_k"],
        K: Float[Tensor, "batch_size ... seq_len d_k"],
        V: Float[Tensor, "batch_size ... seq_len d_v"],
        mask: Bool[Tensor, "batch_size seq_len seq_len"]| None = None
    ) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    if mask is not None:
        attention_mask = -torch.ones_like(mask, dtype=torch.float32)*torch.inf
        attention_mask = torch.where(mask, torch.zeros_like(mask, dtype=torch.float32), attention_mask)
    else:
        attention_mask = torch.ones((Q.shape[-2], K.shape[-2]))
    q_dim_size = torch.tensor(Q.shape[-1])
    q_k = einsum(Q, K, "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2")/torch.sqrt(q_dim_size)
    q_k_softmax = softmax(q_k+attention_mask, dim=-1)
    return einsum(q_k_softmax, V, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: int | None=None, max_seq_len: int | None=None, device=None, dtype=None):
        super().__init__()

        self.num_heads = num_heads
        self.l1 = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.l2 = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_model, max_seq_len, device)
    
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]):
        batch, seq_len, _ = x.shape
        x = self.l1(x)
        Qs, Ks, Vs = x.chunk(chunks=3, dim=-1)
        Qh = torch.stack(Qs.chunk(chunks=self.num_heads, dim=-1), dim=1)
        Kh = torch.stack(Ks.chunk(chunks=self.num_heads, dim=-1), dim=1)
        Vh = torch.stack(Vs.chunk(chunks=self.num_heads, dim=-1), dim=1)
        mask = torch.triu(torch.ones((batch, self.num_heads, seq_len, seq_len))).transpose(-2, -1).to(bool)
        attention = scaled_dot_product_attention(Qh, Kh, Vh, mask)
        attention_concat = rearrange(attention, 'b h seq d_k -> b seq (h d_k)')
        return self.l2(attention_concat)
