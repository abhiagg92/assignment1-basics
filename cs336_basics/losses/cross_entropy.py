import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]):
    x_max = torch.max(inputs, dim=1, keepdim=True).values
    diff = inputs - x_max
    x_exp = torch.exp(diff)
    return -(diff[torch.arange(inputs.shape[0]), targets] - torch.log(x_exp.sum(dim=1))).mean()