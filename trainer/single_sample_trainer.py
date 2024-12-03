import torch
import random
import wandb
from trainer.base_trainer import BaseTrainer
from common.registry import registry

@registry.register_task("single_sample_train")
class SingleSampleTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
    def train(self):
        train_loader, _ = self.load_datasets()
        model = self.load_model()

        max_epochs = self.config["train"].get("max_epochs", 10000)
        tolerance = self.config["train"].get("tolerance", 1e-4)

        criterion  = self.load_criterion()
        optimizer = self.load_optimizer(model.parameters())

        if self.config["train"].get("random_sample", False):
            first_sample = random.choice(list(train_loader))
        else:
            first_sample = next(iter(train_loader))

        inputs, targets = first_sample

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        model.to(self.device)
        print(f"Training on {self.device}.")

        if self.config["wandb"].get("enable_wandb", False):
            wandb.login()
            wandb.init(
                project=self.config["wandb"]["project"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )

        model.train()
        for epoch in range(max_epochs):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            if self.config["wandb"].get("enable_wandb", False):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                })
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{max_epochs}], Loss: {loss.item():.6f}')
            
            if loss.item() < float(tolerance):
                print(f"Loss has reached {loss.item():.6f}, stopping training.")
                return  # Stop training when loss reaches the tolerance
            
        print(f"Reached maximum epochs ({max_epochs}) without achieving tolerance loss.")