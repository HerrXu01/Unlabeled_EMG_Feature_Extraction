import argparse
import yaml
import wandb
import copy
from trainer.base_trainer import Trainer
from trainer.single_sample_trainer import SingleSampleTrainer
from common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="Unlabeled EMG Feature Extraction.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the config YAML file.'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        required=True, 
        choices=['train', 'sweep', 'single_sample_train'], 
        help='Task to run: train or single_sample_train.'
    )
    parser.add_argument(
        '--sweep_config', 
        type=str, 
        required=False, 
        default=None, 
        help='Path to the sweep config YAML file for wandb sweeps (optional).'
    )
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_sweep(experiment_config_origin, sweep_config):
    experiment_config = copy.deepcopy(experiment_config_origin)

    wandb.init(
        config=sweep_config,
        project=experiment_config["wandb"]["project"],
        name=f"SeNic_transformer_sweep_{wandb.util.generate_id()}"
    )

    # Only update experiment_config with parameters from sweep_config['parameters']
    for key, value in sweep_config["parameters"].items():
        if key in wandb.config:  # Ensure it exists in wandb.config
            param_value = wandb.config[key]
            found = False
            for exp_key, exp_value in experiment_config.items():
                if isinstance(exp_value, dict) and key in exp_value:
                    experiment_config[exp_key][key] = param_value
                    found = True
                    break

            if not found:
                print(f"Warning: Parameter '{key}' not found in experiment configuration. Adding it to 'others'.")
                if "others" not in experiment_config:
                    experiment_config["others"] = {}
                experiment_config["others"][key] = param_value

    registry.get_task_class("sweep")(experiment_config).train()

def main():
    args = parse_args()
    experiment_config_origin = load_config(args.config)
    if args.task == "sweep":
        if args.sweep_config is None:
            raise ValueError("Please provide the path to the sweep_config YAML file using the '--sweep_config' argument.")
        sweep_config = load_config(args.sweep_config)
    
    experiment_config_origin["task"] = {"name": args.task}
    
    if experiment_config_origin["wandb"].get("enable_wandb", False):
        wandb.login(force=True)

    if args.task == "sweep":
        sweep_id = wandb.sweep(sweep_config, project=experiment_config_origin["wandb"]["project"])
        print(f"Sweep initialized with ID: {sweep_id}")
        wandb.agent(sweep_id, function=lambda: run_sweep(experiment_config_origin, sweep_config), count=experiment_config_origin["wandb"].get("sweep_count", 20))
    else:
        registry.get_task_class(args.task)(experiment_config_origin).train()

if __name__ == "__main__":
    main()