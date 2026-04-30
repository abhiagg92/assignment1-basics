import os
import torch
from pathlib import Path
import wandb
import numpy as np

from cs336_basics.config import TrainingConfig
from cs336_basics.models import TransformerLM
from cs336_basics.dataloader import get_batch
from cs336_basics.optimizers import AdamW
from cs336_basics.losses import cross_entropy
from cs336_basics.utils.checkpointing import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, config: TrainingConfig):
        self._config = config
        self._batch_size = config.batch_size
        self._context_len = config.context_length
        self._start_iter_num = 0

        self._device = torch.device("cuda")

        self._train_data = np.memmap(config.train_file_path, np.int64, mode="r")
        self._val_data = np.memmap(config.val_file_path, np.int64, mode="r")

        self._model = TransformerLM(
            vocab_size=config.vocab_size,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            theta=config.rope_theta,
            context_length=config.context_length,
        )
        self._model.to(self._device)

        self._optimizer = AdamW(
            self._model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )

        if self._config.resume:
            ckpt_dir = os.path.join(self._config.log_dir, self._config.exp_name)
            ckpt_path = self._get_ckpt_file(ckpt_dir)
            self._start_iter_num = load_checkpoint(ckpt_path, self._model, self._optimizer)

    def train(self):
        num_iters = self._config.num_train_iters
        for i in range(self._start_iter_num, num_iters):
            x, y = get_batch(self._train_data, self._batch_size, self._context_len, self._device)
            self._optimizer.zero_grad()

            preds = self._model(x)
            loss = cross_entropy(preds, y)

            loss.backward()
            self._optimizer.step()

            if i % self._config.ckpt_interval:
                val_loss = self.validate()
                outpath = os.path.join(self._config.log_dir, self._config.exp_name, f"model{i:5d}.pt")
                save_checkpoint(self._model, self._optimizer, i, outpath)
                wandb.log({"train_loss": loss.item(), "val_loss": val_loss})
                self._model.train()

    @torch.no_grad()
    def validate(self):
        self._model.eval()
        total_val_loss = 0
        num_iters = self._config.num_val_iters
        for _ in range(num_iters):
            x, y = get_batch(self._val_data, self._batch_size, self._context_len, self._device)
            preds = self._model(x)
            loss = cross_entropy(preds, y)
            total_val_loss += loss.item()
        return total_val_loss/num_iters


    def _get_ckpt_file(self, ckpt_dir: str):
        ckpt_dir_path = Path(ckpt_dir)
        if self._config.ckpt_name:
            ckpt_path = ckpt_dir_path / self._config.ckpt_name
            if ckpt_path.is_file():
                return str(ckpt_path)
        
        ckpt_files = sorted(ckpt_dir_path.glob("*.pt"))
        return ckpt_files[-1]