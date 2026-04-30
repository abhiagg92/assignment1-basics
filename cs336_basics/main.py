import argparse
import os
import wandb

from cs336_basics.trainer import Trainer
from cs336_basics.config import TrainingConfig


def init_wandb(training_config: TrainingConfig):
    run = wandb.init(
        entity="abhinavaggrwal-na",
        project=training_config.exp_name,
        config=training_config.model_dump()
    )
    return run


def main(args):
    training_config = TrainingConfig.from_file(args.config)
    
    out_path = os.path.join(training_config.log_dir, training_config.exp_name, "config.json")
    training_config.to_json(out_path)
    
    run = init_wandb(training_config)
    trainer = Trainer(training_config)
    trainer.train()

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for transformer training")
    parser.add_argument(
        "--config", required=True, type=str, help="Config file path with training parameters"
    )
    args = parser.parse_args()

    main(args)
