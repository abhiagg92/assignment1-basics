import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        embeddings = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        nn.init.trunc_normal_(embeddings, a=-3, b=3)
        self.embeddings = nn.Parameter(embeddings)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]