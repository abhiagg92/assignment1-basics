import torch
from torch import Tensor, nn
from jaxtyping import Int

from cs336_basics.models import Embedding, Transformer, RMSNorm, Linear


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: int, context_length: int, device=None, dtype=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [Transformer(d_model, num_heads, d_ff, theta, context_length, device, dtype) for i in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    
    def forward(self, x: Int[Tensor, "batch_size sequence_length"]):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x