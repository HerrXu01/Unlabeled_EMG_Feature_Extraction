import torch
from common.registry import registry


@registry.register_optimizer("Adam")
def adam_optim(model_params, lr):
    return torch.optim.Adam(model_params, lr=lr)

@registry.register_optimizer("AdamW")
def adamw_optim(model_params, lr, weight_decay):
    return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)

@registry.register_optimizer("RMSProp")
def rmsprop_optim(model_params, lr):
    return torch.optim.RMSprop(model_params, lr=lr)

@registry.register_optimizer("Nadam")
def nadam_optim(model_params, lr):
    return torch.optim.NAdam(model_params, lr=lr)