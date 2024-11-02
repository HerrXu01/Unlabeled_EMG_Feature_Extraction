import torch
import torch.nn as nn
from common.registry import registry

@registry.register_model("TCN4EMG")
class TCN4EMG(nn.Module):
    def __init__(self, config):
        super(TCN4EMG, self).__init__()
        self.input_dim = config["dataset"]["num_channels"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.output_dim = config["model"]["output_dim"]
        self.num_layers = config["model"].get("num_layers", 4)
        self.kernel_size = config["model"].get("kernel_size", 3)
        self.dropout_prob = config["model"].get("dropout_prob", 0.2)

        layers = []
        dilation = 1
        current_dim = self.input_dim

        for i in range(self.num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=current_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size - 1) * dilation,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_prob))
            layers.append(nn.utils.weight_norm(nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_prob))

            current_dim = self.hidden_dim
            dilation *= 2

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Input x: (batch_size, seq_len, num_channels)
        x = x.permute(0, 2, 1)  # Convert to (batch_size, num_channels, seq_len)
        out = self.network(x)
        out = out[:, :, -1]  # Take the last time step's features
        output = self.fc(out)
        return output
