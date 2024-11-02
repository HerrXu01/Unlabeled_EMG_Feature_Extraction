import argparse
import yaml
import wandb
from trainer.base_trainer import Trainer
from trainer.single_sample_trainer import SingleSampleTrainer
from common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="Unlabeled EMG Feature Extraction.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    parser.add_argument('--task', type=str, required=True, choices=['train', 'single_sample_train'], help='Task to run: train or single_sample_train.')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)

    wandb.login(force=True)

    registry.get_task_class(args.task)(config).train()

if __name__ == "__main__":
    main()