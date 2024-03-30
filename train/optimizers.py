import torch

OPTIMIZERS = {"Adam": torch.optim.Adam,
              "AdamW": torch.optim.AdamW,
              "SGD": torch.optim.SGD,
              "ASGD": torch.optim.ASGD
             }

def get_optimizer(cfg, model_params):
    return OPTIMIZERS[cfg["optimizer"]](model_params)

