import torch
from collections.abc import Iterable
import math


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_g_norm_squared = 0
    for param in parameters:
        if param.grad is None:
            continue
        g_norm = torch.linalg.norm(param.grad)
        total_g_norm_squared += g_norm.item()**2

    total_g_norm = math.sqrt(total_g_norm_squared)
    if total_g_norm >= max_l2_norm:
        scale_factor = max_l2_norm/(total_g_norm+1e-6)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad *= scale_factor