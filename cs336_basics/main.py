import argparse
import wandb
from pathlib import Path

from cs336_basics.trainer import Trainer
from cs336_basics.decoder import Decoder
from cs336_basics.config import TrainingConfig, DecoderConfig


def init_wandb(training_config: TrainingConfig):
    run = wandb.init(
        entity="abhinavaggrwal-na",
        project="cs336_basics",
        config=training_config.model_dump(),
        mode='online'
    )
    return run


def train(args):
    training_config = TrainingConfig.from_file(args.config)
    
    exp_path = Path(training_config.log_dir) / training_config.exp_name
    out_path = exp_path/ "config.json"
    exp_path.mkdir(parents=True, exist_ok=True)
    training_config.to_json(str(out_path))
    
    run = init_wandb(training_config)
    trainer = Trainer(training_config)
    trainer.train()

    run.finish()

def decode(args):
    config = DecoderConfig.from_file(args.config)
    decoder = Decoder(config)
    output = decoder.decode(args.prompt, args.max_tokens)
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for transformer training")
    parser.add_argument(
        "--config", required=True, type=str, help="Config file path with training parameters"
    )
    parser.add_argument(
        "--mode", required=True, type=str, choices=["train", "decode"]
    )
    parser.add_argument("--prompt", type=str, help="Decoder text prompt")
    parser.add_argument("--max_tokens", type=int, help="Max number of tokens generated")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "decode":
        decode(args)
