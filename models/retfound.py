# retfound.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.layers import trunc_normal_

from config.constants import *
from . import models_vit  # from RETFound repo
from utils.pos_embed import interpolate_pos_embed


class RETFoundClassifier(pl.LightningModule):
    """
    RETFound Vision Transformer without BaseClassifier.
    You can use:
        - forward(x) for logits
        - forward_features(x) for backbone features (used in distillation)
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth"),
        drop_path_rate=0.2,
        learning_rate=DIST_LR,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Build ViT backbone (RETFound repo)
        self.model = models_vit.__dict__["vit_large_patch16"](
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=True,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()

        # Remove wrong head weights (shape mismatch)
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Interpolate position embedding if resolution mismatch
        interpolate_pos_embed(self.model, checkpoint_model)

        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys:", msg.missing_keys)

        # re-init classification head
        trunc_normal_(self.model.head.weight, std=2e-5)

    # ---------------------
    #   Forward Passes
    # ---------------------
    def forward(self, x):
        """Returns logits (full classifier forward)."""
        return self.model(x)

    def forward_features(self, x, return_all_tokens=False):
        if return_all_tokens:
            return self.model.forward_features(x)  # returns full sequence (B, N, D)
        return self.model.forward_features(x)[:, 0]  # cls token or pooled

    # ---------------------
    #   Optimizer
    # ---------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # ---------------------
    #   Optional Training Step (NOT used for distillation)
    # ---------------------
    def training_step(self, batch, batch_idx):
        """Only needed if you fine-tune RETFound; otherwise unused."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Only needed if you fine-tune RETFound; otherwise unused."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss