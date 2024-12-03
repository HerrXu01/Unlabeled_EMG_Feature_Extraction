import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from common.registry import registry


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )



class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["model"]["n_embed"] % config["model"]["n_head"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config["model"]["n_embed"], 3 * config["model"]["n_embed"], bias=config["model"]["bias"])
        # output projection
        self.c_proj = nn.Linear(config["model"]["n_embed"], config["model"]["n_embed"], bias=config["model"]["bias"])
        # regularization
        self.attn_dropout = nn.Dropout(config["model"]["dropout"])
        self.resid_dropout = nn.Dropout(config["model"]["dropout"])
        self.n_head = config["model"]["n_head"]
        self.n_embd = config["model"]["n_embed"]
        self.dropout = config["model"]["dropout"]
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config["window"]["window_size"], config["window"]["window_size"])).view(
                    1, 1,config["window"]["window_size"], config["window"]["window_size"]
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["model"]["n_embed"], 4 * config["model"]["n_embed"], bias=config["model"]["bias"])
        self.c_proj = nn.Linear(4 * config["model"]["n_embed"], config["model"]["n_embed"], bias=config["model"]["bias"])
        self.dropout = nn.Dropout(config["model"]["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["model"]["n_embed"], bias=config["model"]["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["model"]["n_embed"], bias=config["model"]["bias"])
        self.mlp = MLP(config)

    def forward(self, x):
        x_1 = self.ln_1(x)
        x_1 = self.attn(x_1)
        x = x + x_1
        x_2 = self.ln_2(x)
        x_2 = self.mlp(x_2)
        x = x + x_2
        return x

    
class OneHot(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        x = F.one_hot(x.long(), self.output_size)
        return torch.flatten(x.float(), start_dim=-2)  # flatten as 1-D vector


class SumTokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config["model"]["vocab_size"], config["model"]["n_embed"]) for _ in range(8)]
        )

    def forward(self, x):
        # B, T, C
        x = [self.embeddings[i](x[:, :, i].long()) for i in range(8)]
        return torch.sum(torch.stack(x, dim=-1), dim=-1)


class ConcatTokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config["model"]["n_embed"] % 8 == 0
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config["model"]["vocab_size"], config["model"]["n_embed"] // 8) for _ in range(8)]
        )

    def forward(self, x):
        # B, T, C
        x = [self.embeddings[i](x[:, :, i].long()) for i in range(8)]
        return torch.cat(x, dim=-1)


class TokenEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        token_embedding_type = config["model"]["token_embedding_type"]
        if token_embedding_type == "basic":
            self.embedding = nn.Embedding(config["model"]["vocab_size"], config["model"]["n_embed"])
        elif token_embedding_type == "FC":
            self.embedding = nn.Linear(8, config["model"]["n_embed"], bias=False)
        elif (
             token_embedding_type == "FC_extended"
        ):  # first convert to one-hot encoding
            self.embedding = nn.Sequential(
                OneHot(config["model"]["vocab_size"]),
                nn.Linear(config["model"]["vocab_size"] * 8, config["model"]["n_embed"], bias=False),
            )
        elif token_embedding_type == "basic_sum":
            self.embedding = SumTokenEmbedding(config)
        elif token_embedding_type == "basic_concat":
            self.embedding = ConcatTokenEmbedding(config)

    def forward(self, x):
        return self.embedding(x)


class GPT_base(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["window"].get("window_size", None) is not None
        assert config["model"].get("vocab_size", None) is not None
        self.config = config

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        raise NotImplementedError

@registry.register_model("ChatEMG")
class ChatEMG(GPT_base):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = self.config["model"]["vocab_size"]
        self.window_size = self.config["window"]["window_size"] - 1
        self.n_embed = self.config["model"]["n_embed"]
        self.dropout = self.config["model"].get("dropout", 0.0)
        self.n_layer = self.config["model"]["n_layer"]
        self.num_channels = self.config["dataset"]["num_channels"]

        self.transformer_channel = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, self.n_embed),
                wpe=nn.Embedding(self.window_size, self.n_embed),
                drop=nn.Dropout(self.dropout),
                h=nn.ModuleList([Block(config) for _ in range(self.n_layer)]),
                ln_f=LayerNorm(self.n_embed, bias=self.config["model"]["bias"]),
            )
        )

        self.transformer_context = nn.ModuleDict(
            dict(
                wte=TokenEmbedding(config),
                wpe=nn.Embedding(self.window_size, self.n_embed),
                drop=nn.Dropout(self.dropout),
                h=nn.ModuleList([Block(config) for _ in range(self.n_layer)]),
                ln_f=LayerNorm(self.n_embed, bias=self.config["model"]["bias"]),
            )
        )

        self.latent_decoder = nn.Linear(self.n_embed * 2, self.n_embed, bias=False)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)
        self.transformer_channel.wte.weight = (
            self.lm_head.weight
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
                )
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer_context.wpe.weight.numel()
            n_params -= self.transformer_channel.wpe.weight.numel()
        return n_params

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()[0], idx.size()[1]
        assert (
            t <= self.window_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.window_size}"

        print(f"The datatype of inputs is {idx.dtype}.")

        predictions = []

        for selected_channel in range(self.num_channels):
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1,t)

            tok_emb_context = self.transformer_context.wte(
                idx.float()
            )  # token embeddings of shape (b, t, n_embd)
            pos_emb_context = self.transformer_context.wpe(
                pos
            )  # position embeddings of shape (1, t, n_embd)

            x_context = self.transformer_context.drop(tok_emb_context + pos_emb_context)

            for block in self.transformer_context.h:
                x_context = block(x_context)
            x_context = self.transformer_context.ln_f(x_context)

            tok_emb_channel = self.transformer_channel.wte(
                idx[:, :, selected_channel].long()
            ).reshape(
                b, t, self.n_embed
            )  # token embeddings of shape (b, t, n_embd)
            pos_emb_channel = self.transformer_channel.wpe(
                pos
            )  # position embeddings of shape (1, t, n_embd)

            x_channel = self.transformer_channel.drop(tok_emb_channel + pos_emb_channel)

            for block in self.transformer_channel.h:
                x_channel = block(x_channel)
            x_channel = self.transformer_channel.ln_f(x_channel)

            # Extract the last time step's output
            x_context_last = x_context[:, -1, :]  # shape (batch_size, n_embed)
            x_channel_last = x_channel[:, -1, :]  # shape (batch_size, n_embed)

            # Concatenate and pass through latent decoder
            x_combined = torch.cat((x_channel_last, x_context_last), dim=1)  # shape (batch_size, n_embed * 2)
            x = self.latent_decoder(x_combined)  # shape (batch_size, n_embed)

            # Pass through lm_head to get the final output
            x = self.lm_head(x)  # shape (batch_size, vocab_size)

            predictions.append(x)
            

        return torch.stack(predictions, dim=1)  # shape (batch_size, num_channels, vocab_size)

