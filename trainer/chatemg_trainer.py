import torch
import wandb
from trainer.base_trainer import BaseTrainer
from models.chatemg import ChatEMG
from torchinfo import summary
from tqdm import tqdm


class ChatEMGTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        train_loader, val_loader = self.load_datasets(
            convert_dtype=self.config["dataset"]["convert_dtype"],
            enable_offset=self.config["dataset"]["enable_offset"],
            offset=self.config["dataset"]["offset"]
        )
        model = self.load_model()
        criterion = self.load_criterion()
        optimizer = self.load_optimizer(model.parameters())
        scheduler = self.load_lr_scheduler(optimizer)

        input_size = (
            self.config["train"]["batch_size"],
            self.config["window"]["window_size"] - 1,
            self.config["dataset"]["num_channels"]
        )
        summary(model, input_size=input_size, dtypes=[torch.long])

        print(f"Training on {self.device}.")

        num_epochs = self.config["train"]["num_epochs"]
        best_val_loss =  float("inf")
        best_model_path = self.config["train"].get("best_model_path", None)
        best_model_state_dict = None
        best_epoch = 0

        if self.config["wandb"].get("enable_wandb", False) and self.config["task"]["name"] != "sweep":
            wandb.init(
                project=self.config["wandb"]["project"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_huber = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                loss = criterion(outputs_flat, targets_flat)
                # To ensure a fair comparison, the Huber loss is calculated here 
                # solely for evaluation purposes and does not participate in optimization.
                huber = self.estimate_huber(outputs_flat, targets_flat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_huber += huber.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            train_huber /= len(train_loader.dataset)

            val_loss, val_huber = self.evaluate(model, val_loader, criterion)

            if self.config["wandb"].get("enable_wandb", False):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss_nll": train_loss,
                    "val_loss_nll": val_loss,
                    "train_huber": train_huber,
                    "val_huber": val_huber,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()
                best_epoch = epoch + 1

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Huber: {train_huber:.4f}, Val Huber: {val_huber:.4f}')

            if scheduler is not None:
                scheduler.step()

        print(f"The best val loss is {best_val_loss}, achieved at Epoch {best_epoch}.")
        if self.config["wandb"].get("enable_wandb", False):
            wandb.log({"best_val_loss": best_val_loss, "best_epoch": best_epoch})
            wandb.finish()
        
        if self.config["train"].get("save_checkpoint", False):
            self.save_checkpoint(best_model_state_dict, best_model_path)



    def evaluate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        val_huber = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                loss = criterion(outputs_flat, targets_flat)
                huber = self.estimate_huber(outputs_flat, targets_flat)
                val_loss += loss.item() * inputs.size(0)
                val_huber += huber.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_huber /= len(val_loader.dataset)

        return val_loss, val_huber

    def estimate_huber(self, logits_flat, targets_flat):
        _, pred = torch.max(logits_flat, dim=1)
        pred = pred.float()
        targets_flat = targets_flat.float()

        criterion = torch.nn.SmoothL1Loss()
        huber = criterion(pred, targets_flat)

        return huber
                


