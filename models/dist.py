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

        # determine whether student will use masking during training by default
        self.mask_student_in_training = getattr(self.student, "mask_ratio", 0.0) > 0.0

    # ---------------------------
    # helper: obtain teacher pooled features aligned to ids_keep if provided
    # ---------------------------
    @torch.no_grad()
    def _teacher_features_aligned(self, x, ids_keep=None):
        """
        If ids_keep is provided and teacher can return tokens, gather teacher tokens at ids_keep and pool;
        otherwise fall back to pooled teacher features.
        """
        # Try to get token-level teacher output
        if ids_keep is None:
            # no alignment requested -> return pooled teacher features
            t = self.teacher.forward_features(x, return_pooled=True)
            return t

        # ids_keep is (B, N_vis)
        try:
            t_seq = self.teacher.forward_features(x, return_pooled=False, return_tokens=True)
        except TypeError:
            # model.forward_features may not accept the kwargs — try without them
            try:
                t_seq = self.teacher.forward_features(x)
                # if returns pooled, t_seq will be 2D -> fallback
                if hasattr(t_seq, "ndim") and t_seq.ndim == 2:
                    return t_seq
                # if model returned tuple, attempt to pick tokens
                if isinstance(t_seq, tuple):
                    found = None
                    for item in t_seq:
                        if hasattr(item, "ndim") and item.ndim == 3:
                            found = item
                            break
                    if found is None:
                        # fallback to pooled interpretation
                        for item in t_seq[::-1]:
                            if hasattr(item, "ndim") and item.ndim == 2:
                                return item
                        return t_seq[0]
                    t_seq = found
            except Exception:
                # give up: return pooled features
                t_pooled = self.teacher.forward_features(x)
                return t_pooled

        # If we have token-level teacher output, gather visible tokens and pool
        if hasattr(t_seq, "ndim") and t_seq.ndim == 3:
            B, N, D_t = t_seq.shape
            # ids_keep indices live on the student's device; ensure they are on same device as t_seq
            ids_keep = ids_keep.to(t_seq.device)

            # gather teacher visible tokens
            t_visible = torch.gather(
                t_seq,
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, D_t)
            )
            t_pooled = t_visible.mean(dim=1)
            return t_pooled

        # fallback: treat t_seq as pooled 2D features
        return t_seq

    # ---------------------------------------------------
    # TRAINING STEP
    # ---------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, _, _ = batch

        # ensure x is on same device as model params
        device = next(self.student.parameters()).device
        if x.device != device:
            x = x.to(device)

        # -------------------------
        # Student (apply mask according to student.mask_ratio if enabled)
        # -------------------------
        apply_mask = self.mask_student_in_training
        s, mask, ids_keep, ids_restore = self.student.forward_features(
            x,
            return_pooled=False,
            apply_mask=apply_mask
        )
        # s = (B, N, D) sequence (full-length after mask tokens inserted)
        B, N, D = s.shape

        # Pool student over visible tokens only
        if ids_keep is None:
            s_pooled = s.mean(dim=1)  # full sequence pooling
        else:
            # ids_keep refers to positions of visible tokens in the full sequence
            visible = torch.gather(
                s,
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
            )
            s_pooled = visible.mean(dim=1)  # (B, D)

        # -------------------------
        # Teacher (aligned pooling if student was masked)
        # -------------------------
        with torch.no_grad():
            if ids_keep is None:
                t = self.teacher.forward_features(x, return_pooled=True)
            else:
                # align teacher pooling to student visible tokens when possible
                t = self._teacher_features_aligned(x, ids_keep=ids_keep)

        # -------------------------
        # Project & cosine loss
        # -------------------------
        s_proj = self.projector(s_pooled)
        # ensure shapes match
        if s_proj.shape != t.shape:
            # if teacher returns different dim, try to coerce
            # move t to same device as s_proj
            t = t.to(s_proj.device)
            if s_proj.shape[1] != t.shape[1]:
                raise RuntimeError(f"Feature dim mismatch: student_proj={s_proj.shape}, teacher={t.shape}")

        cosine = F.cosine_similarity(s_proj, t, dim=1)
        distill_loss = (1 - cosine).mean()

        self.log("train/distill_loss", distill_loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return distill_loss

    # ---------------------------------------------------
    # VALIDATION STEP
    # ---------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, _, _ = batch

        device = next(self.student.parameters()).device
        if x.device != device:
            x = x.to(device)

        # Use unmasked student for validation by default (more stable), but if you prefer to
        # validate masked student set apply_mask=True here.
        apply_mask = False

        s, mask, ids_keep, ids_restore = self.student.forward_features(
            x,
            return_pooled=False,
            apply_mask=apply_mask
        )

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

        with torch.no_grad():
            if ids_keep is None:
                t = self.teacher.forward_features(x, return_pooled=True)
            else:
                t = self._teacher_features_aligned(x, ids_keep=ids_keep)

        if s_proj.shape != t.shape:
            t = t.to(s_proj.device)
            if s_proj.shape[1] != t.shape[1]:
                raise RuntimeError(f"Feature dim mismatch: student_proj={s_proj.shape}, teacher={t.shape}")

        cosine = F.cosine_similarity(s_proj, t, dim=1)
        distill_loss = (1 - cosine).mean()

        self.log("val/distill_loss", distill_loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return distill_loss

    # ---------------------------------------------------
    # OPTIMIZER + WARMUP + COSINE ANNEALING
    # ---------------------------------------------------
    def configure_optimizers(self):
        params = list(self.student.parameters()) + list(self.projector.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        total_epochs = self.trainer.max_epochs  # ← dynamic
        warmup_epochs = 10
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
