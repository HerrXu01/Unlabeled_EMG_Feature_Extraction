import torch
from common.registry import registry


@registry.register_criterion("MSE")
def mse_loss():
    return torch.nn.MSELoss()

@registry.register_criterion("L1")
def l1_loss():
    return torch.nn.L1Loss()

@registry.register_criterion("Huber")
def huber_loss():
    return torch.nn.SmoothL1Loss()

@registry.register_criterion("LogCosh")
def log_cosh_loss():
    class LogCoshLoss(torch.nn.Module):
        def forward(self, y_pred, y_true):
            return torch.mean(torch.log(torch.cosh(y_pred - y_true)))
    return LogCoshLoss()
