import torch
import numpy.typing as npt
import numpy as np
import random


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    slice_indices = [random.randrange(len(dataset)-context_length) for _ in range(batch_size)]
    input_batch, target_batch = [], []
    for idx in slice_indices:
        input = dataset[idx:idx+context_length]
        target = dataset[idx+1:idx+1+context_length]
        input_batch.append(input)
        target_batch.append(target)

    input_batch_tensor = torch.tensor(np.array(input_batch), dtype=torch.long, device=device)
    target_batch_tensor = torch.tensor(np.array(target_batch), dtype=torch.long, device=device)
    return input_batch_tensor, target_batch_tensor