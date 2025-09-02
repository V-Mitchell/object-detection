import torch
from utils.dict import remove_key

OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "ASGD": torch.optim.ASGD,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

SCHEDULERS = {
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def get_optimizer(cfg, model_params):
    return OPTIMIZERS[cfg["type"]](model_params, **remove_key(cfg, ["type"]))


def get_scheduler(cfg, optimizer):
    return SCHEDULERS[cfg["type"]](optimizer, **remove_key(cfg, ["type"]))
