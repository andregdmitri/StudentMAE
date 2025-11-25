"""
Fixed vmamba_backbone.py — VimBlock corrected to be paper-faithful to Algorithm 1

Key fixes applied compared to previous copy:
- B and C heads now produce per-expanded-channel SSM params with shape (B, M, E, N)
  (heads output expand * ssm_dim and are reshaped). Delta scaling preserved.
- ssm_recurrence accepts C with shape (B, M, E, N) and uses it directly (no broadcasting
  of a shared C across E).
- z is linear; SiLU applied only at gating time.
- Depthwise convs and log-space prefix-product remain for numerical stability.

This file is intended as a drop-in replacement for the original module.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.constants import *

class VimBlock(nn.Module):
    """
    Paper-faithful VimBlock (Algorithm 1: Vision Mamba).

    - d_model: input token dimension D
    - expand: expanded channel dimension E
    - ssm_dim: SSM internal dimension N
    - kernel_size: Conv1d kernel size for per-channel conv

    Notes:
    - Heads B and C now produce (B, M, E, N) by outputting expand * ssm_dim and
      reshaping. Delta still scales B if desired to match Alg.1 line 14 behavior.
    - z is linear; gating uses SiLU(z) only at merge time.
    - ssm_recurrence is vectorized and implemented in log-space for improved stability.
    """

    def __init__(self, d_model: int, expand: int, ssm_dim: int, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.expand = expand  # E
        self.ssm_dim = ssm_dim  # N

        # normalization
        self.norm = nn.LayerNorm(d_model)

        # project token -> x (E) and gating z (E)
        self.to_x = nn.Linear(d_model, expand)
        self.to_z = nn.Linear(d_model, expand)

        # conv1d: depthwise per-channel conv to match per-expanded-channel conv behavior
        pad = (kernel_size - 1) // 2
        self.conv_f = nn.Conv1d(expand, expand, kernel_size, padding=pad, groups=expand)
        self.conv_b = nn.Conv1d(expand, expand, kernel_size, padding=pad, groups=expand)

        # SSM parameter heads for forward/backward:
        # B, C will produce (B, M, E * N) then be reshaped to (B, M, E, N)
        self.head_B_f = nn.Linear(expand, expand * ssm_dim)
        self.head_C_f = nn.Linear(expand, expand * ssm_dim)
        self.head_Delta_f = nn.Linear(expand, expand)
        self.paramA_f = nn.Parameter(torch.randn(expand, ssm_dim) * 0.01)

        self.head_B_b = nn.Linear(expand, expand * ssm_dim)
        self.head_C_b = nn.Linear(expand, expand * ssm_dim)
        self.head_Delta_b = nn.Linear(expand, expand)
        self.paramA_b = nn.Parameter(torch.randn(expand, ssm_dim) * 0.01)

        # learnable ParameterDelta/bias terms (one per expanded channel E)
        # Alg.1: Delta_o <- softplus( LinearDelta(x') + ParameterDelta_o )
        self.delta_bias_f = nn.Parameter(torch.zeros(expand))
        self.delta_bias_b = nn.Parameter(torch.zeros(expand))

        # final projection back to d_model
        self.to_out = nn.Linear(expand, d_model)

        self._init_weights()

    def _init_weights(self):
        # encourage small initial Δ changes
        nn.init.zeros_(self.head_Delta_f.weight)
        nn.init.zeros_(self.head_Delta_f.bias)
        nn.init.zeros_(self.head_Delta_b.weight)
        nn.init.zeros_(self.head_Delta_b.bias)
        # small init for paramA already done at creation

    def ssm_recurrence(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized SSM recurrence, faithful to:
            h_t = A_t * h_{t-1} + B_t * x_t    (elementwise over N)
            y_t = sum_N ( h_t * C_t )

        Inputs:
            A: (B, M, E, N)
            B: (B, M, E, N)
            C: (B, M, E, N)
            x: (B, M, E)
        Returns:
            y: (B, M, E)

        Implementation detail:
            We compute prefix products P_t = prod_{i=0..t} A_i in log-space for stability.
        """
        Bsz, M, E, N = A.shape
        device = A.device
        dtype = A.dtype
        eps = 1e-6

        # Separate sign and magnitude to compute stable prefix products
        signA = torch.sign(A)
        absA = torch.clamp(torch.abs(A), min=eps)

        # log-space cumulative product of absolute values
        log_absA = torch.log(absA)  # (B, M, E, N)
        logP = torch.cumsum(log_absA, dim=1)  # (B, M, E, N)
        P_abs = torch.exp(logP)  # (B, M, E, N)

        # sign prefix product
        signP = torch.cumprod(signA, dim=1)  # (B, M, E, N)

        # full prefix product
        P = signP * P_abs  # (B, M, E, N)

        # invP used in term formulation: invP_t = 1 / P_t  (avoid divide by zero)
        invP = 1.0 / (P + eps)

        # prepare x as (B, M, E, 1)
        x_unsq = x.unsqueeze(-1)  # (B, M, E, 1)

        # term_k = invP_k * B_k * x_k -> (B, M, E, N)
        term = invP * B * x_unsq

        # S_t = cumsum(term, dim=1)
        S = torch.cumsum(term, dim=1)  # (B, M, E, N)

        # h_t = P_t * S_t
        h = P * S  # (B, M, E, N)

        # y_t = sum_N ( h_t * C_t ) -> (B, M, E)
        y = (h * C).sum(dim=-1)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, D)
        returns: (B, M, D) after residual (Alg.1 line 28)
        """
        Bsz, M, D = x.shape

        x_norm = self.norm(x)                # (B, M, D)
        x_proj = self.to_x(x_norm)           # (B, M, E)
        # z must be linear per Alg.1; SiLU applied when gating.
        z = self.to_z(x_norm)                # (B, M, E)

        # conv expects (B, C, L) so transpose, conv, transpose back
        xf = self.conv_f(x_proj.transpose(1, 2)).transpose(1, 2)  # (B, M, E)
        xf = F.silu(xf)

        xb = self.conv_b(x_proj.flip(dims=[1]).transpose(1, 2)).transpose(1, 2).flip(dims=[1])
        xb = F.silu(xb)

        # produce SSM params
        # Heads produce (B, M, E * N) then reshape -> (B, M, E, N)
        Bf = self.head_B_f(xf).view(Bsz, M, self.expand, self.ssm_dim)   # (B, M, E, N)
        Cf = self.head_C_f(xf).view(Bsz, M, self.expand, self.ssm_dim)   # (B, M, E, N)
        Delta_f = F.softplus(self.head_Delta_f(xf) + self.delta_bias_f.view(1, 1, -1))  # (B, M, E)
        # Af shape (B, M, E, N) = Delta_f.unsqueeze(-1) * paramA_f (E, N)
        Af = Delta_f.unsqueeze(-1) * self.paramA_f.unsqueeze(0).unsqueeze(0)  # (B, M, E, N)
        # Optionally scale B by Delta (Alg.1 line 14): keep for compatibility
        Bf_full = Delta_f.unsqueeze(-1) * Bf  # (B, M, E, N)

        Bb = self.head_B_b(xb).view(Bsz, M, self.expand, self.ssm_dim)
        Cb = self.head_C_b(xb).view(Bsz, M, self.expand, self.ssm_dim)
        Delta_b = F.softplus(self.head_Delta_b(xb) + self.delta_bias_b.view(1, 1, -1))
        Ab = Delta_b.unsqueeze(-1) * self.paramA_b.unsqueeze(0).unsqueeze(0)
        Bb_full = Delta_b.unsqueeze(-1) * Bb

        # run SSM recurrence for forward/backward
        y_f = self.ssm_recurrence(Af, Bf_full, Cf, xf)   # (B, M, E)
        y_b = self.ssm_recurrence(Ab, Bb_full, Cb, xb)   # (B, M, E)

        # gated merge, paper: y' = y * SiLU(z) (apply SiLU to z at gating time only)
        g = F.silu(z)  # (B, M, E)
        gated_f = y_f * g
        gated_b = y_b * g

        y = gated_f + gated_b  # (B, M, E)

        out = self.to_out(y)  # (B, M, D)
        return out + x


class VisualMamba(nn.Module):
    """Vision Mamba (Vim) style backbone compatible with previous API."""

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
        use_cls_token: bool = True,
        variant="tiny",
        **kwargs,
    ):

        super().__init__()

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.use_cls_token = use_cls_token

        # patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # number of patches
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # optional class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter('cls_token', None)

        # positional embedding (num_patches + cls?)
        n_pos = num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_pos, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # mask token for MAE-style masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # backbone: stack VimBlocks
        self.backbone = nn.ModuleList([
            VimBlock(d_model=embed_dim, expand=expand_dim, ssm_dim=ssm_dim)
            for _ in range(depth)
        ])

        # final norm
        self.norm = nn.LayerNorm(embed_dim)

        # init weights for patch_embed
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    # ---------------------
    # Masking utilities (compatible with original)
    # ---------------------
    def apply_mask(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MAE-style masking adapted for tokens including optional cls token."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, N, D = x.shape

        # If no masking required
        if mask_ratio <= 0.0:
            mask = torch.ones(B, N, device=x.device)
            ids_keep = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_restore = ids_keep
            return x, mask, ids_keep, ids_restore

        has_cls = self.use_cls_token
        start = 1 if has_cls else 0
        num_vis_tokens = N - start

        noise = torch.rand(B, num_vis_tokens, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        num_mask = int(math.floor(num_vis_tokens * mask_ratio))
        ids_mask = ids_shuffle[:, :num_mask]
        ids_keep = ids_shuffle[:, num_mask:]

        # shift indices to account for cls token if present
        if has_cls:
            ids_mask = ids_mask + 1
            ids_keep = ids_keep + 1

        # build visible tokens (keep cls if present)
        if has_cls:
            cls_tok = x[:, 0:1, :]
            x_vis = torch.gather(x[:, start:, :], dim=1, index=(ids_keep - start).unsqueeze(-1).expand(-1, -1, D))
            x_keep = torch.cat([cls_tok, x_vis], dim=1)
        else:
            x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask binary vector (1=keep, 0=masked)
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_mask, 0)

        # mask tokens to append
        mask_tokens = self.mask_token.expand(B, num_mask + (1 if has_cls else 0), D)
        x_combined = torch.cat([x_keep, mask_tokens[:, :num_mask, :]], dim=1)

        if has_cls:
            full_ids = torch.cat([torch.zeros(B, 1, dtype=ids_keep.dtype, device=ids_keep.device),
                                  (ids_keep - 1),
                                  (ids_mask - 1)], dim=1)
            ids_combined_positions = torch.arange(x_combined.size(1), device=x.device).unsqueeze(0).expand(B, -1)
            ids_restore = torch.empty(B, N, dtype=ids_combined_positions.dtype, device=x.device)
            for b in range(B):
                orig_positions = full_ids[b]
                ids_restore[b].scatter_(0, orig_positions, ids_combined_positions[b])
        else:
            full_ids = torch.cat([ids_keep, ids_mask], dim=1)
            ids_combined_positions = torch.arange(x_combined.size(1), device=x.device).unsqueeze(0).expand(B, -1)
            ids_restore = torch.empty(B, N, dtype=ids_combined_positions.dtype, device=x.device)
            for b in range(B):
                ids_restore[b].scatter_(0, full_ids[b], ids_combined_positions[b])

        x_full = torch.gather(x_combined, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        return x_full, mask, ids_keep, ids_restore

    # ---------------------
    # Forwarding utilities
    # ---------------------
    def forward_features(self, img: torch.Tensor, return_pooled: bool = True, apply_mask: bool = False):
        B = img.shape[0]

        # patch embedding
        x = self.patch_embed(img)             # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)

        # optionally add class token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)           # (B, N+1, D)

        # add positional embeddings
        x = x + self.pos_embed

        # optionally mask
        if apply_mask and self.mask_ratio > 0.0:
            x_seq, mask, ids_keep, ids_restore = self.apply_mask(x)
        else:
            x_seq = x
            mask, ids_keep, ids_restore = None, None, None

        # pass through backbone (ModuleList)
        for blk in self.backbone:
            x_seq = blk(x_seq)

        x_seq = self.norm(x_seq)

        if return_pooled:
            if self.use_cls_token:
                pooled = x_seq[:, 0]
            else:
                pooled = x_seq.mean(dim=1)

            if apply_mask and ids_keep is not None:
                Bsz, Nfull, D = x_seq.shape
                visible = torch.gather(x_seq, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
                return visible.mean(dim=1)

            return pooled

        return x_seq, mask, ids_keep, ids_restore

    def forward(self, x: torch.Tensor):
        return self.forward_features(x, return_pooled=True, apply_mask=False)
