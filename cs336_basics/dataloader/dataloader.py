import torch
import numpy.typing as npt
import numpy as np


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    slice_indices = np.random.choice(len(dataset)-context_length, batch_size, replace=False)
    input_batch, target_batch = [], []
    for idx in slice_indices:
        input = dataset[idx:idx+context_length]
        target = dataset[idx+1:idx+1+context_length]
        input_batch.append(input)
        target_batch.append(target)

    input_batch_tensor = torch.LongTensor(np.array(input_batch), device=device)
    target_batch_tensor = torch.LongTensor(np.array(target_batch), device=device)
    return input_batch_tensor, target_batch_tensor