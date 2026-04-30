import json
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    exp_name: str

    vocab_size: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: int
    context_length: int

    num_train_iters: int
    num_val_iters: int
    batch_size: int

    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float

    log_dir: str
    ckpt_interval: int

    resume: bool
    ckpt_name: str | None

    train_file_path: str
    val_file_path: str

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as f:
            config = json.load(f)   
        return cls(**config)
    
    def to_json(self, out_path: str):
        training_config = self.model_dump_json(indent=4)
        with open(out_path, "w") as f:
            f.write(training_config)

class DecoderConfig(BaseModel):
    vocab_size: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: int
    context_length: int

    num_train_iters: int
    num_val_iters: int
    batch_size: int

    vocab_filepath: str
    merges_filepath: str
    special_tokens: list[str]

    top_p: float
    temp: float