import torch
import wandb
import os
import datetime
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from models.lstm import LSTM4EMG
from models.tcn import TCN4EMG
from models.transformer import Transformer4EMG
from models.chatemg import ChatEMG
from data_processing.datasets import EMGDataset
from data_processing.preprocess import EMGPreprocessor
from sklearn.model_selection import train_test_split
from common.registry import registry
import trainer.criterion
import trainer.optimizer
from tqdm import tqdm

@registry.register_task("train")
@registry.register_task("sweep")
class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.config["train"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_self_supervised_pairs(self, windows):
        """
        Split each window into an input sequence and a target sequence for self-supervised learning.
        
        Args:
            windows (np.ndarray): Windowed data with shape (num_windows, window_size, num_channels).
        
        Returns:
            inputs (np.ndarray): Input sequences (num_windows, window_size - 1, num_channels).
            targets (np.ndarray): Target sequences (num_windows, num_channels).
        """
        inputs = windows[:, :-1, :]
        targets = windows[:, -1, :]
        return inputs, targets

    def split_data(self, inputs, targets, test_size=0.2, shuffle=True):
        """
        Split the input-output pairs into training and validation sets.
        
        Args:
            inputs (np.ndarray): Input sequences.
            targets (np.ndarray): Target sequences.
            test_size (float): Proportion of the data to be used as validation set.
        
        Returns:
            train_inputs, val_inputs, train_targets, val_targets: Split training and validation sets.
        """
        return train_test_split(inputs, targets, test_size=test_size, random_state=42, shuffle=shuffle)

    def load_datasets(self):
        """
        Prepare dataloader for training and validation.
        """
        windows_dir = self.config["window"].get("windows_dir", None)
        filename = self.config["window"]["filename"]
        batch_size = self.config["train"]["batch_size"]
        shuffle = self.config["train"]["enable_shuffle"]

        windows = None
        if windows_dir:
            file_path = os.path.join(windows_dir, filename)
            if os.path.exists(windows_dir) and os.path.isfile(file_path):
                print(f"Sliding windows data found at: {file_path}")
                print("Loading data from existing files ...")
                windows = np.load(file_path)

        if windows is None:    
            print("Processing data ...")
            data_processor = EMGPreprocessor(self.config)
            windows = data_processor.process()

        inputs, targets = self.create_self_supervised_pairs(windows)
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)
        train_inputs, val_inputs, train_targets, val_targets = self.split_data(
            inputs,
            targets,
            test_size=self.config["train"]["val_size"],
            shuffle=shuffle
        )

        train_dataset = EMGDataset(train_inputs, train_targets)
        val_dataset = EMGDataset(val_inputs, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def load_model(self):
        model = registry.get_model_class(
            self.config["model"]["name"]
        )(self.config)

        return model.to(self.device)

    def load_criterion(self):
        criterion = registry.get_criterion(
            self.config["train"]["criterion"]
        )()

        return criterion

    def load_optimizer(self, model_params):
        optim_params = {
            "model_params": model_params,
            "lr": self.config["train"]["learning_rate"]
        }
        weight_decay = self.config["train"].get("weight_decay", None)
        if weight_decay is not None:
            optim_params["weight_decay"] = weight_decay

        return registry.get_optimizer(self.config["train"]["optimizer"])(**optim_params)

    def load_lr_scheduler(self, optimizer):
        if self.config["train"].get("enable_lr_decay", False):
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config["train"].get("lr_decay_step", 10),
                gamma=self.config["train"].get("lr_decay_gamma", 0.1)
            )
            return scheduler
        return None

    def train(self):
        train_loader, val_loader = self.load_datasets()
        model = self.load_model()
        criterion = self.load_criterion()
        optimizer = self.load_optimizer(model.parameters())
        scheduler = self.load_lr_scheduler(optimizer)

        input_size = (
            self.config["train"]["batch_size"],
            self.config["window"]["window_size"] - 1,
            self.config["dataset"]["num_channels"]
        )
        summary(model, input_size=input_size)

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
            for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)

            val_loss = self.evaluate(model, val_loader, criterion)

            if self.config["wandb"].get("enable_wandb", False):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()
                best_epoch = epoch + 1

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

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
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        return val_loss / len(val_loader.dataset)

    def save_checkpoint(self, state_dict, best_model_path):
        if state_dict is None:
            return
        best_model_path = best_model_path or "checkpoints/best_model.pth"
        timestamp = self.config["train"]["timestamp"]
        best_model_path = best_model_path.replace(".pth", f"_{timestamp}.pth")
        checkpoint_dir = os.path.dirname(best_model_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state_dict, best_model_path)



