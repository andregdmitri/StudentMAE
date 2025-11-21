# vmamba_backbone.py

from typing import Optional, Tuple
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from config.constants import *

class VisualMamba(nn.Module):
    """
    Visual Mamba:
    - patch_embed conv
    - Mamba backbone (sequence of Mamba blocks)
    - forward_features used for distillation (returns embedding of size embed_dim)
    - head: linear classifier (num_classes)
    """

    def __init__(
        self,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        ssm_dim=SSM_DIM,
        expand_dim=EXPAND_DIM,
        learning_rate=DIST_LR,
        mask_ratio=MASK_RATIO,
        variant="tiny",
        **mamba_kwargs,
    ):

        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate


        # 1. Patch Embedding Layer
        # This layer converts an image into a sequence of flattened patch embeddings.
        # For example, a 224x224 image with 16x16 patches becomes 14x14 = 196 patches.
        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Mamba Backbone (L blocks)
        # A sequence of Mamba blocks to process the patch embeddings.
        self.backbone = nn.Sequential(
            *[
                Mamba(
                    d_model=embed_dim,   # D
                    d_state=ssm_dim,     # N
                    expand=expand_dim,   # E
                    **mamba_kwargs
                )
                for _ in range(depth)
            ]
        )

        # 3. Classification Headed
        # A normalization layer and a linear layer to produce the final logits.
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)


    def apply_mask(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MAE-style masking applied to sequence `x`.

        Args:
            x: (B, N, D) sequence
            mask_ratio: override self.mask_ratio if provided
        Returns:
            x_masked: (B, N_visible, D) — student input (only kept tokens; masked tokens optionally appended as mask tokens)
            mask: (B, N) binary mask where 0 indicates masked token (for visualization/metrics)
            ids_keep: (B, N_visible) indices of kept tokens (so teacher can align)
            ids_restore: (B, N) indices to restore original order (not strictly needed here)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, N, D = x.shape

        if mask_ratio <= 0.0:
            mask = torch.ones(B, N, device=x.device)
            ids_keep = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_restore = ids_keep
            return x, mask, ids_keep, ids_restore

        # -------------------------------
        # 1. Random shuffle
        # -------------------------------
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # -------------------------------
        # 2. Split keep vs mask indices
        # -------------------------------
        num_mask = int(N * mask_ratio)
        ids_mask = ids_shuffle[:, :num_mask]
        ids_keep = ids_shuffle[:, num_mask:]

        # visible tokens
        x_keep = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # -------------------------------
        # 3. Build binary mask (1=keep, 0=masked)
        # -------------------------------
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_mask, 0)

        # -------------------------------
        # 4. INSERT MASK TOKENS (MAE FIX)
        # -------------------------------
        mask_tokens = self.mask_token.expand(B, num_mask, D)

        # MAE concatenation: [visible tokens | mask tokens]
        x_combined = torch.cat([x_keep, mask_tokens], dim=1)

        # -------------------------------
        # 5. Restore original token order
        # -------------------------------
        x_full = torch.gather(
            x_combined,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        return x_full, mask, ids_keep, ids_restore

    def forward_features(self, img: torch.Tensor, return_pooled: bool = True, apply_mask: bool = False):
        """Return pooled features by default (B, D). If return_pooled=False returns sequence + mask info.

        If apply_mask=True, uses self.apply_mask and returns x_keep + mask info.
        """
        # patch embed
        x = self.patch_embed(img)          # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)

        # ---------------------------
        # Apply proper MAE-style masking
        # ---------------------------
        if apply_mask and self.mask_ratio > 0.0:
            # now returns full-length masked sequence (B, N, D)
            x_seq, mask, ids_keep, ids_restore = self.apply_mask(x)
        else:
            x_seq = x
            mask, ids_keep, ids_restore = None, None, None

        # ---------------------------
        # Backbone expects (B, N, D)
        # ---------------------------
        x_seq = self.backbone(x_seq)       # (B, N, D)
        x_seq = self.norm(x_seq)

        # ---------------------------
        # Output pooled or full sequence
        # ---------------------------
        if return_pooled:

            if apply_mask and ids_keep is not None:
                # Pool only visible tokens (MAE style)
                B, N, D = x_seq.shape
                visible = torch.gather(
                    x_seq,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
                )
                return visible.mean(dim=1)

            # Normal full-sequence pooling
            return x_seq.mean(dim=1)

        return x_seq, mask, ids_keep, ids_restore

    def forward(self, x: torch.Tensor):
        # default: return pooled features for classifier compatibility
        return self.forward_features(x, return_pooled=True, apply_mask=False)
