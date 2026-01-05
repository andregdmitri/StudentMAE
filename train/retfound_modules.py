# train/retfound_modules.py

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import *
from models.retfound import RETFoundClassifier


# -------------------------------------------------------------------
# Backbone loader
# -------------------------------------------------------------------
def load_retfound_backbone(path=None):

    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")

    full = RETFoundClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=path,
    ).eval()

    vit = full.model
    embed_dim = vit.head.weight.shape[1]
    return vit, embed_dim


def get_feats(backbone, x):
    with torch.no_grad():
        return backbone.forward_features(x)


# -------------------------------------------------------------------
# Linear Probe Module
# -------------------------------------------------------------------
class RETFoundLinearProbe(pl.LightningModule):
    def __init__(self, backbone, embed_dim, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.eval()

        self.head = nn.Linear(embed_dim, NUM_CLASSES)

        if class_weights is not None:
            self.register_buffer("cw", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.cw)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        feats = get_feats(self.backbone, x)
        return self.head(feats)

    def _shared_step(self, batch):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return loss, preds, probs, y

    def training_step(self, batch, _):
        loss, preds, probs, y = self._shared_step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss, preds, probs, y = self._shared_step(batch)
        self.acc.update(preds, y)
        self.f1.update(preds, y)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc", self.acc.compute())
        self.log("val/f1", self.f1.compute())
        self.acc.reset()
        self.f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=self.hparams.lr)


# -------------------------------------------------------------------
# Fine-Tune Module
# -------------------------------------------------------------------
class RETFoundFineTune(RETFoundLinearProbe):
    def __init__(self, backbone, embed_dim, lr, class_weights=None):
        super().__init__(backbone, embed_dim, lr, class_weights)

        # UNFREEZE backbone
        for p in self.backbone.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [{"params": self.backbone.parameters(), "lr": self.hparams.lr * 0.1},
             {"params": self.head.parameters(), "lr": self.hparams.lr}],
            weight_decay=1e-4,
        )
