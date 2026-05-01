import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    sum_inv = 1/x_exp.sum(dim=dim, keepdim=True)
    return x_exp*sum_inv