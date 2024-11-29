"""
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
        self.positional_encoding = self._get_positional_encoding(self.window_size, self.N_embed).float()

       # Feature Expansion Layer
        self.feature_expansion = nn.Linear(1, self.N_embed)
        self.channel_expansion = nn.Conv1d(self.num_channels, self.N_embed, kernel_size=1)
        
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
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(length, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x shape: (batch_size, window_size, num_channels)
        device = x.device  # Ensure all tensors are on the same device
        batch_size = x.size(0)
        predictions = []
        
        ##### 修改位置编码的形状 #####
        # Move positional encoding to the same device as input
        positional_encoding = self.positional_encoding.unsqueeze(0).to(device)  # shape: (1, window_size, N_embed)
        
        ##### 修改通道扩展部分的实现 #####
        # Expand feature dimension to N_embed for all channels together
        x = x.permute(0, 2, 1).to(torch.float32)  # shape: (batch_size, num_channels, window_size)
        x = self.channel_expansion(x).to(torch.float32)  # Ensure the output is float32  # shape: (batch_size, N_embed, window_size)
        x = x.permute(0, 2, 1)  # shape: (batch_size, window_size, N_embed)
        
        # Add positional encoding
        x += positional_encoding
        
        # Apply Layer Normalization (optional)
        if self.enable_norm:
            x = self.layer_norm(x)
        
        # Pass through Context Encoder
        context_encoded = self.context_encoder(x.permute(1, 0, 2))  # shape: (window_size, batch_size, N_embed)
        context_encoded = context_encoded.permute(1, 0, 2)  # shape: (batch_size, window_size, N_embed)
        
        ##### 将 Layer Normalization 应用到循环之外 #####
        # Apply Layer Normalization to context_encoded (optional)
        if self.enable_norm:
            context_encoded = self.context_layer_norm(context_encoded)
        
        ##### 修改单个通道的编码实现 #####
        # Loop over each channel to predict its next value (using original input x)
        original_x = x.permute(0, 2, 1)  # shape: (batch_size, num_channels, window_size)
        for i in range(self.num_channels):
            # Extract features for the current channel i
            channel_i_features = original_x[:, i, :].unsqueeze(2).to(torch.float32)  # shape: (batch_size, window_size, 1)
            
            # Expand feature dimension to N_embed
            channel_i_features = self.feature_expansion(channel_i_features).to(torch.float32)  # shape: (batch_size, window_size, N_embed)
            
            # Add positional encoding
            channel_i_features += positional_encoding
            
            # Apply Layer Normalization (optional)
            if self.enable_norm:
                channel_i_features = self.layer_norm(channel_i_features)
            
            # Pass through Single Channel Encoder
            channel_i_encoded = self.single_channel_encoder(channel_i_features.permute(1, 0, 2))  # shape: (window_size, batch_size, N_embed)
            channel_i_encoded = channel_i_encoded.permute(1, 0, 2)  # shape: (batch_size, window_size, N_embed)
            
            # Concatenate Encodings
            combined_encoding = torch.cat((channel_i_encoded, context_encoded), dim=-1)  # shape: (batch_size, window_size, 2 * N_embed)
            
            # Fully Connected Layers to Predict Next Value for Channel i
            output = self.fc_layers(combined_encoding)  # shape: (batch_size, window_size, 1)
            
            # Take the last time step as the predicted value for channel i at next time point
            predictions.append(output[:, -1, :])  # shape: (batch_size, 1)
        
        # Stack all predictions to get the final output
        predictions = torch.cat(predictions, dim=1)  # shape: (batch_size, num_channels)
        
        return predictions
"""


import torch
import torch.nn as nn
from common.registry import registry

@registry.register_model("Transformer4EMG")
class Transformer4EMG(nn.Module):
    def __init__(self, config):
        super(Transformer4EMG, self).__init__()
        
        # 获取模型超参数
        self.N_embed = config["model"].get("N_embed", 128)
        self.num_attention_heads = config["model"].get("num_attention_heads", 8)
        self.num_encoder_layers = config["model"].get("num_encoder_layers", 2)
        self.dropout_rate = config["model"].get("dropout_prob", 0.1)
        self.num_channels = config["dataset"]["num_channels"]
        self.window_size = config["window"]["window_size"] - 1
        self.enable_norm = config["model"].get("enable_norm", True)
        self.channels_weight_share = config["model"].get("channels_weight_share", True)
        
        # 位置编码
        self.positional_encoding = self._get_positional_encoding(self.window_size, self.N_embed).float()
        
        # 特征扩展层
        if self.channels_weight_share:
            self.feature_expansion = nn.Linear(1, self.N_embed)
        else:
            self.feature_expansions = nn.ModuleList([nn.Linear(1, self.N_embed) for _ in range(self.num_channels)])
        
        # 通道扩展层
        self.channel_expansion = nn.Conv1d(self.num_channels, self.N_embed, kernel_size=1)
        
        # 单通道编码器
        if self.channels_weight_share:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.N_embed,
                nhead=self.num_attention_heads,
                dim_feedforward=self.N_embed * 4,
                dropout=self.dropout_rate,
                activation='relu'
            )
            self.single_channel_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        else:
            self.single_channel_encoders = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.N_embed,
                        nhead=self.num_attention_heads,
                        dim_feedforward=self.N_embed * 4,
                        dropout=self.dropout_rate,
                        activation='relu'
                    ),
                    num_layers=self.num_encoder_layers
                ) for _ in range(self.num_channels)
            ])
        
        # 上下文编码器（所有通道一起）
        encoder_layer_context = nn.TransformerEncoderLayer(
            d_model=self.N_embed,
            nhead=self.num_attention_heads,
            dim_feedforward=self.N_embed * 4,
            dropout=self.dropout_rate,
            activation='relu'
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer_context, num_layers=self.num_encoder_layers)
        
        # 层归一化
        if self.enable_norm:
            self.context_layer_norm = nn.LayerNorm(self.N_embed)
            if self.channels_weight_share:
                self.layer_norm = nn.LayerNorm(self.N_embed)
            else:
                self.layer_norms = nn.ModuleList([nn.LayerNorm(self.N_embed) for _ in range(self.num_channels)])
        
        # 预测的全连接层
        if self.channels_weight_share:
            self.fc_layers = nn.Sequential(
                nn.Linear(self.N_embed * 2, self.N_embed * 2),
                nn.LayerNorm(self.N_embed * 2) if self.enable_norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(self.N_embed * 2, self.N_embed),
                nn.LayerNorm(self.N_embed) if self.enable_norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(self.N_embed, 1)
            )
        else:
            self.fc_layers_list = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.N_embed * 2, self.N_embed * 2),
                    nn.LayerNorm(self.N_embed * 2) if self.enable_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(self.N_embed * 2, self.N_embed),
                    nn.LayerNorm(self.N_embed) if self.enable_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(self.N_embed, 1)
                ) for _ in range(self.num_channels)
            ])
    
    def _get_positional_encoding(self, length, d_model):
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(length, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x 的形状: (batch_size, window_size, num_channels)
        device = x.device  # 确保所有张量在同一设备上
        batch_size = x.size(0)
        predictions = []
        
        # 位置编码
        positional_encoding = self.positional_encoding.unsqueeze(0).to(device)  # 形状: (1, window_size, N_embed)
        
        # 通道扩展
        x_channels = x.permute(0, 2, 1).to(torch.float32)  # 形状: (batch_size, num_channels, window_size)
        x_expanded = self.channel_expansion(x_channels).to(torch.float32)  # 形状: (batch_size, N_embed, window_size)
        x_expanded = x_expanded.permute(0, 2, 1)  # 形状: (batch_size, window_size, N_embed)
        
        # 添加位置编码
        x_expanded += positional_encoding
        
        # 层归一化（可选）
        if self.enable_norm and self.channels_weight_share:
            x_expanded = self.layer_norm(x_expanded)
        
        # 上下文编码器
        context_encoded = self.context_encoder(x_expanded.permute(1, 0, 2))  # 形状: (window_size, batch_size, N_embed)
        context_encoded = context_encoded.permute(1, 0, 2)  # 形状: (batch_size, window_size, N_embed)
        
        # 上下文层归一化（可选）
        if self.enable_norm:
            context_encoded = self.context_layer_norm(context_encoded)
        
        # 遍历每个通道
        for i in range(self.num_channels):
            # 提取当前通道的特征
            channel_i_data = x[:, :, i].unsqueeze(2).to(torch.float32)  # 形状: (batch_size, window_size, 1)
            
            if self.channels_weight_share:
                # 特征扩展
                channel_i_features = self.feature_expansion(channel_i_data)
                # 添加位置编码
                channel_i_features += positional_encoding
                # 层归一化（可选）
                if self.enable_norm:
                    channel_i_features = self.layer_norm(channel_i_features)
                # 单通道编码器
                channel_i_encoded = self.single_channel_encoder(channel_i_features.permute(1, 0, 2))
                channel_i_encoded = channel_i_encoded.permute(1, 0, 2)
                # 拼接编码
                combined_encoding = torch.cat((channel_i_encoded, context_encoded), dim=-1)
                # 预测下一时刻的值
                output = self.fc_layers(combined_encoding)
            else:
                # 特征扩展（使用独立的层）
                channel_i_features = self.feature_expansions[i](channel_i_data)
                # 添加位置编码
                channel_i_features += positional_encoding
                # 层归一化（可选，使用独立的层）
                if self.enable_norm:
                    channel_i_features = self.layer_norms[i](channel_i_features)
                # 单通道编码器（使用独立的编码器）
                channel_i_encoded = self.single_channel_encoders[i](channel_i_features.permute(1, 0, 2))
                channel_i_encoded = channel_i_encoded.permute(1, 0, 2)
                # 拼接编码
                combined_encoding = torch.cat((channel_i_encoded, context_encoded), dim=-1)
                # 预测下一时刻的值（使用独立的全连接层）
                output = self.fc_layers_list[i](combined_encoding)
            
            # 取最后一个时间步的输出作为预测值
            predictions.append(output[:, -1, :])  # 形状: (batch_size, 1)
        
        # 拼接所有预测值
        predictions = torch.cat(predictions, dim=1)  # 形状: (batch_size, num_channels)
        
        return predictions
