import torch
import torch.nn as nn
import torch.nn.functional as F
from common.registry import registry

@registry.register_model("Transformer4EMG")
class Transformer4EMG(nn.Module):
    def __init__(self, config):
        super(Transformer4EMG, self).__init__()
        
        # Get model hyperparameters from config
        self.N_embed = config["model"].get("N_embed", 128)
        self.num_attention_heads = config["model"].get("num_attention_heads", 8)
        self.num_encoder_layers = config["model"].get("num_encoder_layers", 2)
        self.dropout_rate = config["model"].get("dropout_prob", 0.1)
        self.num_channels = config["dataset"]["num_channels"]
        self.window_size = config["window"]["window_size"] - 1
        self.enable_norm = config["model"].get("enable_norm", True)
        
        # Positional Encoding
        self.positional_encoding = self._get_positional_encoding(self.window_size, self.N_embed)

        # Feature Expansion Layer
        self.feature_expansion = nn.Linear(1, self.N_embed)
        self.channel_expansion = nn.Linear(self.num_channels, self.N_embed)
        
        # Single Channel Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.N_embed,
            nhead=self.num_attention_heads,
            dim_feedforward=self.N_embed * 4,
            dropout=self.dropout_rate,
            activation='relu'
        )
        self.single_channel_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        
        # Context Encoder (All channels together)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        
        # Layer Normalization
        if self.enable_norm:
            self.layer_norm = nn.LayerNorm(self.N_embed)
            self.context_layer_norm = nn.LayerNorm(self.N_embed)
        
        # Fully Connected Layers for Prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.N_embed * 2, self.N_embed * 2),
            nn.LayerNorm(self.N_embed * 2) if self.enable_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.N_embed * 2, self.N_embed),
            nn.LayerNorm(self.N_embed) if self.enable_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.N_embed, 1)
        )
        
    def _get_positional_encoding(self, length, d_model):
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x shape: (batch_size, window_size, num_channels)
        device = x.device  # Ensure all tensors are on the same device
        batch_size = x.size(0)
        predictions = []
        
        # Move positional encoding to the same device as input
        positional_encoding = self.positional_encoding.unsqueeze(0).to(device)  # shape: (1, window_size, N_embed)
        
        # Expand feature dimension to N_embed for all channels together
        x = x.permute(0, 2, 1)  # shape: (batch_size, num_channels, window_size)
        x = self.channel_expansion(x).permute(0, 2, 1)  # shape: (batch_size, window_size, N_embed)
        
        # Add positional encoding
        x += positional_encoding
        
        # Apply Layer Normalization
        if self.enable_norm:
            x = self.layer_norm(x)
        
        # Pass through Context Encoder
        context_encoded = self.context_encoder(x.permute(1, 0, 2)).to(device)  # shape: (window_size, batch_size, N_embed)
        context_encoded = context_encoded.permute(1, 0, 2)  # shape: (batch_size, window_size, N_embed)
        
        # Apply Layer Normalization to context_encoded
        if self.enable_norm:
            context_encoded = self.context_layer_norm(context_encoded)
        
        # Loop over each channel to predict its next value (using original input x)
        original_x = x.permute(0, 2, 1)  # shape: (batch_size, num_channels, window_size)
        for i in range(self.num_channels):
            # Extract features for the current channel i
            channel_i_features = original_x[:, i, :].unsqueeze(2)  # shape: (batch_size, window_size, 1)
            
            # Expand feature dimension to N_embed
            channel_i_features = self.feature_expansion(channel_i_features).to(device)  # shape: (batch_size, window_size, N_embed)
            
            # Add positional encoding
            channel_i_features += positional_encoding
            
            # Apply Layer Normalization (optional)
            if self.enable_norm:
                channel_i_features = self.layer_norm(channel_i_features)
            
            # Pass through Single Channel Encoder
            channel_i_encoded = self.single_channel_encoder(channel_i_features.permute(1, 0, 2)).to(device)  # shape: (window_size, batch_size, N_embed)
            channel_i_encoded = channel_i_encoded.permute(1, 0, 2)  # shape: (batch_size, window_size, N_embed)
            
            # Concatenate Encodings
            combined_encoding = torch.cat((channel_i_encoded, context_encoded), dim=-1)  # shape: (batch_size, window_size, 2 * N_embed)
            
            # Fully Connected Layers to Predict Next Value for Channel i
            output = self.fc_layers(combined_encoding).to(device)  # shape: (batch_size, window_size, 1)
            
            # Take the last time step as the predicted value for channel i at next time point
            predictions.append(output[:, -1, :])  # shape: (batch_size, 1)
        
        # Stack all predictions to get the final output
        predictions = torch.cat(predictions, dim=1)  # shape: (batch_size, num_channels)
        
        return predictions
