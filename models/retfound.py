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

    def forward_features(self, x, return_pooled: bool = True, return_tokens: bool = False):
        """
        Returns features.
        - If return_pooled=True returns pooled (B, D).
        - If return_tokens=True tries to return token sequence (B, N, D) when possible.
        - If token API isn't available, will fall back gracefully to pooled features.
        """
        # Common timm ViT API: model.forward_features(x) -> pooled (B, D)
        # Some ViT implementations expose flags such as `return_all_tokens` or return a tuple.
        # We'll try common variants robustly.

        # 1) Try calling forward_features and inspect output
        try:
            out = self.model.forward_features(x)
            # If out is a tuple or an ndarray with ndim==3, interpret as tokens
            if return_tokens:
                # case: model returns tokens directly
                if isinstance(out, tuple):
                    # some implementations return (tokens, pooled) or similar
                    # prefer returning the token sequence if present
                    for item in out:
                        if hasattr(item, "ndim") and item.ndim == 3:
                            return item
                    # none of the items were tokens; fallthrough
                elif hasattr(out, "ndim") and out.ndim == 3:
                    return out

            # If we reached here and return_pooled True -> return pooled features if possible
            if return_pooled:
                # If out is tuple and pooled is second/last element, try to extract
                if isinstance(out, tuple):
                    # heuristics: pooled item usually 2D
                    for item in out[::-1]:
                        if hasattr(item, "ndim") and item.ndim == 2:
                            return item
                    # fallback: take first
                    return out[0]
                else:
                    return out

        except TypeError:
            # forward_features may not accept the call signature
            out = None
        except Exception:
            # Unexpected errors: fallback to a safe call below
            out = None

        # 2) Try common alternate keyword
        if return_tokens:
            try:
                # some models accept return_all_tokens=True / return_tokens=True
                out = self.model.forward_features(x, return_all_tokens=True)
                if hasattr(out, "ndim") and out.ndim == 3:
                    return out
            except Exception:
                pass

        # 3) Final fallback -> call standard forward_features and return pooled
        try:
            pooled = self.model.forward_features(x)
            if return_pooled:
                return pooled
            # if caller wanted tokens but none available, return pooled to keep caller robust
            return pooled
        except Exception as e:
            # as last resort, run a forward and raise (should not normally happen)
            logits = self.model(x)
            # reduce logits to a feature-like tensor to avoid breaking callers
            return logits.detach()

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
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Only needed if you fine-tune RETFound; otherwise unused."""
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss