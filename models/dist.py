import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision

from config.constants import NUM_CLASSES


class DistillationModule(pl.LightningModule):
    """
    Improved Visual Feature Distillation Module:
      • Student learns to match teacher features via cosine similarity
      • Student head is trained with CE loss for meaningful supervised metrics
      • Supports masking inside student.forward_features(apply_mask=True)
    """

    def __init__(self, teacher, student, projector, lr, distill_weight=1.0, ce_weight=0.2):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher", "student", "projector"])

        self.teacher = teacher.eval()
        self.student = student
        self.projector = projector

        self.lr = lr
        self.distill_weight = distill_weight
        self.ce_weight = ce_weight

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # -----------------------------
        # Metrics (weighted → handles class imbalance safely)
        # -----------------------------
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.train_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.train_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.train_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES, average="weighted")

        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.val_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES, average="weighted")

    # ---------------------------------------------------
    # TRAINING STEP
    # ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch

        # ---------- Teacher forward ----------
        with torch.no_grad():
            t = self.teacher.forward_features(x)        # (B, Tdim)

        # ---------- Student forward ----------
        # IMPORTANT: Use student masking if student.mask_ratio > 0
        s = self.student.forward_features(x, apply_mask=True)   # (B, Sdim)
        s_proj = self.projector(s)

        # ---------- Distillation Loss ----------
        cosine = F.cosine_similarity(s_proj, t, dim=1)
        distill_loss = (1 - cosine).mean()

        # ---------- Supervised CE Loss for classification ----------
        logits = self.student.head(s)
        ce_loss = F.cross_entropy(logits, y)

        # ---------- Total Loss ----------
        loss = self.distill_weight * distill_loss + self.ce_weight * ce_loss

        # ---------- Metrics ----------
        preds_proba = torch.softmax(logits, dim=1)
        preds = preds_proba.argmax(dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)
        self.train_auroc.update(preds_proba, y)
        self.train_aupr.update(preds_proba, y)

        # Logging
        self.log("loss/distill", distill_loss, on_epoch=True, prog_bar=True)
        self.log("loss/ce", ce_loss, on_epoch=True, prog_bar=False)
        self.log("loss/total", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ---------------------------------------------------
    # VALIDATION STEP
    # ---------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch

        with torch.no_grad():
            t = self.teacher.forward_features(x)
            s = self.student.forward_features(x, apply_mask=True)
            s_proj = self.projector(s)

            cosine = F.cosine_similarity(s_proj, t, dim=1)
            distill_loss = (1 - cosine).mean()

            logits = self.student.head(s)
            ce_loss = F.cross_entropy(logits, y)

        loss = self.distill_weight * distill_loss + self.ce_weight * ce_loss

        preds_proba = torch.softmax(logits, dim=1)
        preds = preds_proba.argmax(dim=1)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_auroc.update(preds_proba, y)
        self.val_aupr.update(preds_proba, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/distill_loss", distill_loss, on_epoch=True)
        self.log("val/ce_loss", ce_loss, on_epoch=True)

        return loss

    # ---------------------------------------------------
    # END-OF-EPOCH METRIC LOGGING
    # ---------------------------------------------------
    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), prog_bar=True)
        self.log("train/f1", self.train_f1.compute(), prog_bar=True)
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=False)
        self.log("train/aupr", self.train_aupr.compute(), prog_bar=False)

        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()
        self.train_aupr.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=False)
        self.log("val/aupr", self.val_aupr.compute(), prog_bar=False)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()

    # ---------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------
    def configure_optimizers(self):
        params = list(self.student.parameters()) + list(self.projector.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # --- Warmup + Cosine Annealing ---
        warmup_epochs = 10
        total_epochs = 50
        cosine_epochs = total_epochs - warmup_epochs
        base_lr = self.lr                # e.g., 5e-4
        final_lr = 1e-6

        def lr_lambda(epoch):
            # ---- linear warmup for first 10 epochs ----
            if epoch < warmup_epochs:
                return epoch / warmup_epochs

            # ---- cosine annealing for remaining epochs ----
            progress = (epoch - warmup_epochs) / cosine_epochs
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
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

