import torch.nn as nn
from common.registry import registry

"""
@registry.register_model("LSTM4EMG")
class LSTM4EMG(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        dropout_prob=0.5,
        enable_norm=False
    ):
        super(LSTM4EMG, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.enable_norm = enable_norm
        if self.enable_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.enable_norm:
            lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output
"""

@registry.register_model("LSTM4EMG")
class LSTM4EMG(nn.Module):
    def __init__(self, config):
        super(LSTM4EMG, self).__init__()
        self.input_dim = config["dataset"]["num_channels"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.output_dim = config["model"]["output_dim"]
        self.num_layers = config["model"].get("num_layers", 1)
        self.dropout_prob = config["model"].get("dropout_prob", 0)
        self.enable_norm = config["model"].get("enable_norm", False)
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0
        )
        if self.enable_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.enable_norm:
            lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output