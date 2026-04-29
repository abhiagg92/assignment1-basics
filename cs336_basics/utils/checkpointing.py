import os
import torch
from typing import BinaryIO, IO
from torch import nn, optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "iteration": iteration
    }
    torch.save(state, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO,
    model: nn.Module,
    optimizer: optim.Optimizer
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]