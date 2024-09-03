import torch

OPTIMIZERS = {"Adam": torch.optim.Adam,
              "AdamW": torch.optim.AdamW,
              "SGD": torch.optim.SGD,
              "ASGD": torch.optim.ASGD
             }

def get_optimizer(optimizer_type, model_params):
    return OPTIMIZERS[optimizer_type](model_params)

