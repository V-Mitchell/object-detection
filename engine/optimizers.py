import torch
from utils.dict import remove_key

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "ASGD": torch.optim.ASGD
}


def get_optimizer(cfg, model_params):
    return OPTIMIZERS[cfg["type"]](model_params, **remove_key(cfg, ["type"]))
