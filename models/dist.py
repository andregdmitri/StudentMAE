# models/dist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

class DistillationModule(pl.LightningModule):
    """
    Phase I: Pure feature distillation.
    - Student learns to match teacher features using cosine similarity
    - No classifier head training
    - No CE loss, no supervised metrics
    """

    def __init__(self, teacher, student, projector, lr):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher", "student", "projector"])

        self.teacher = teacher.eval()      # frozen teacher
        self.student = student             # student encoder (with masking)
        self.projector = projector         # align embed dims if necessary
        self.lr = lr

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    # ---------------------------------------------------
    # TRAINING STEP
    # ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, _ = batch

        # -------------------------
        # Teacher (frozen)
        # -------------------------
        with torch.no_grad():
            t = self.teacher.forward_features(x)      # (B, D)

        # -------------------------
        # Student (masked sequence)
        # -------------------------
        s, mask, ids_keep, ids_restore = self.student.forward_features(
            x,
            return_pooled=False,
            apply_mask=True
        )
        # s = (B, N, D) restored sequence
        # ids_keep = indices of visible tokens

        # -------------------------
        # MAE-style pooling: ONLY visible tokens
        # -------------------------
        B, N, D = s.shape
        if ids_keep is None:
            # mask_ratio = 0 → pooled over the entire sequence
            s_pooled = s.mean(dim=1)
        else:
            visible = torch.gather(
                s,
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )
            s_pooled = visible.mean(dim=1)     # (B, D)

        # -------------------------
        # Project & cosine loss
        # -------------------------
        s_proj = self.projector(s_pooled)
        cosine = F.cosine_similarity(s_proj, t, dim=1)
        distill_loss = (1 - cosine).mean()

        self.log("train/distill_loss", distill_loss, on_epoch=True, prog_bar=True)
        return distill_loss

    # ---------------------------------------------------
    # VALIDATION STEP
    # ---------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # -------------------------
        # Teacher
        # -------------------------
        with torch.no_grad():
            t = self.teacher.forward_features(x)

            # -------------------------
            # Student (masked)
            # -------------------------
            s, mask, ids_keep, ids_restore = self.student.forward_features(
                x,
                return_pooled=False,
                apply_mask=True
            )

            # -------------------------
            # MAE-style pooling: ONLY visible tokens
            # -------------------------
            B, N, D = s.shape

            if ids_keep is None:
                s_pooled = s.mean(dim=1)
            else:
                visible = torch.gather(
                    s,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
                )
                s_pooled = visible.mean(dim=1)

            s_proj = self.projector(s_pooled)

        cosine = F.cosine_similarity(s_proj, t, dim=1)
        distill_loss = (1 - cosine).mean()

        self.log("val/distill_loss", distill_loss, on_epoch=True, prog_bar=True)
        return distill_loss

    # ---------------------------------------------------
    # OPTIMIZER + WARMUP + COSINE ANNEALING
    # ---------------------------------------------------
    def configure_optimizers(self):
        params = list(self.student.parameters()) + list(self.projector.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        warmup_epochs = 10
        total_epochs = 50
        cosine_epochs = total_epochs - warmup_epochs

        base_lr = self.lr
        final_lr = 1e-6

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs

            # ---- cosine annealing for remaining epochs ----
            progress = (epoch - warmup_epochs) / cosine_epochs
            cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
            min_factor = final_lr / base_lr
            return cosine_decay * (1 - min_factor) + min_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
