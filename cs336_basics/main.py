import argparse
import os

from cs336_basics.trainer import Trainer
from cs336_basics.training_config import TrainingConfig


def main(args):
    training_config = TrainingConfig.from_file(args.config)
    
    out_path = os.path.join(training_config.log_dir, training_config.exp_name, "config.json")
    training_config.to_json(out_path)
    
    trainer = Trainer(training_config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for transformer training")
    parser.add_argument(
        "--config", required=True, type=str, help="Config file path with training parameters"
    )
    args = parser.parse_args()

    main(args)
