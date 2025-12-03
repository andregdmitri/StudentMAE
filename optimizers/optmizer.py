# optimizers/optimizer.py

import math
import torch
from torch.optim import AdamW

from config.constants import *

def warmup_cosine_optimizer(
    parameters,
    max_epochs: int,
    lr: float = LR,
    warmup_epochs: int = WARMUP_EPOCHS,
    final_lr: float = FINAL_LR,
    weight_decay: float = WEIGHT_DECAY,
):      
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)

    # number of epochs after warmup
    cosine_epochs = max_epochs - warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs

        progress = (epoch - warmup_epochs) / max(1, cosine_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        min_factor = final_lr / lr

        return cosine_decay * (1 - min_factor) + min_factor

    scheduler = {
        "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
        "interval": "epoch",
        "frequency": 1,
        "name": "warmup_cosine"
    }

    return optimizer, scheduler
