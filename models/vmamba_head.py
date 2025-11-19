# vmamba_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class HeadTrainer(pl.LightningModule):
    def __init__(self, backbone, num_classes, lr):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()         # freeze features
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Linear(backbone.embed_dim, num_classes)
        self.lr = lr

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.head.parameters(), lr=self.lr)