import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from config.constants import *

# Inverse softplus helper (to initialize bias so softplus(target) ~= target)
def inverse_softplus(x: float):
    # numerically stable inverse of softplus
    return math.log(math.expm1(x + 1e-12) + 1e-12)

class VimBlock(nn.Module):
    """
    Vision Mamba (Vim) block — faithful implementation of Algorithm 1 (bidirectional SSM).
    - Bidirectional conv1d + separate SSM heads (forward/backward)
    - Δ parameterization: softplus(LinearΔ(x') + delta_bias)
    - Ao constructed as: Ao = Δo * ParameterAo  (keeps ParameterAo in fp32)
    - Uses selective_scan_fn when available, with a safe python fallback.
    """

    def __init__(self, d_model: int, expand: int, ssm_dim: int, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.expand = expand  # E
        self.ssm_dim = ssm_dim  # N

        # LayerNorm + projections
        self.norm = nn.LayerNorm(d_model)
        self.to_x = nn.Linear(d_model, expand)  # Linear_x
        self.to_z = nn.Linear(d_model, expand)  # Linear_z

        # Forward / Backward Conv1d (depthwise conv per paper)
        pad = (kernel_size - 1) // 2
        self.conv_fwd = nn.Conv1d(expand, expand, kernel_size, padding=pad, groups=expand)
        self.conv_bwd = nn.Conv1d(expand, expand, kernel_size, padding=pad, groups=expand)

        # Heads for forward direction
        self.B_fwd = nn.Linear(expand, expand * ssm_dim)
        self.C_fwd = nn.Linear(expand, expand * ssm_dim)
        self.D_fwd = nn.Linear(expand, expand)          # produces Δ pre-activation (per E)
        self.DeltaBias_fwd = nn.Parameter(torch.zeros(expand))

        # Heads for backward direction
        self.B_bwd = nn.Linear(expand, expand * ssm_dim)
        self.C_bwd = nn.Linear(expand, expand * ssm_dim)
        self.D_bwd = nn.Linear(expand, expand)
        self.DeltaBias_bwd = nn.Parameter(torch.zeros(expand))

        # Learnable base A parameters (per direction) with shape (E, N)
        # Keep as fp32 param — we'll cast carefully later when building Ao
        self.ParameterA_fwd = nn.Parameter(torch.randn(expand, ssm_dim) * -0.5)
        self.ParameterA_bwd = nn.Parameter(torch.randn(expand, ssm_dim) * -0.5)
        # mark them to avoid weight decay if desired
        self.ParameterA_fwd._no_weight_decay = True
        self.ParameterA_bwd._no_weight_decay = True

        # output projection + final residual mapping
        self.to_out = nn.Linear(expand, d_model)

        # init dt biases to inverse-softplus(target_dt) (so softplus yields ~target_dt at init)
        target_dt = 0.01
        inv = inverse_softplus(target_dt)
        # set bias values (as in many SSM designs)
        nn.init.constant_(self.DeltaBias_fwd, inv)
        nn.init.constant_(self.DeltaBias_bwd, inv)

        self.reset_parameters()

    def reset_parameters(self):
        # small initialization for B/C heads to avoid blowup
        nn.init.normal_(self.B_fwd.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.C_fwd.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.B_fwd.bias)
        nn.init.zeros_(self.C_fwd.bias)

        nn.init.normal_(self.B_bwd.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.C_bwd.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.B_bwd.bias)
        nn.init.zeros_(self.C_bwd.bias)

        # small init for D heads
        nn.init.zeros_(self.D_fwd.weight)
        nn.init.zeros_(self.D_fwd.bias)
        nn.init.zeros_(self.D_bwd.weight)
        nn.init.zeros_(self.D_bwd.bias)

        # output projection
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    # ---- selective scan wrapper (tries fast kernel then python fallback) ----
    def _selective_scan(self, A, B, C, X):
        """
        A,B,C: (B, M, E, N)
        X: (B, M, E)
        returns: Y (B, M, E)
        Implementation tries selective_scan_fn (fast) on fp32 and falls back to safe python loop.
        """
        orig_dtype = X.dtype
        Bsz, M, E, N = A.shape

        # ensure contiguous and cast to float32 for the kernel
        A_ = A.permute(0, 2, 1, 3).contiguous().float()
        B_ = B.permute(0, 2, 1, 3).contiguous().float()
        C_ = C.permute(0, 2, 1, 3).contiguous().float()
        X_ = X.permute(0, 2, 1).contiguous().float()

        try:
            Y = selective_scan_fn(A_, B_, C_, X_)  # expected (B, E, M)
            Y = Y.permute(0, 2, 1)  # -> (B, M, E)
            return Y.to(orig_dtype)
        except Exception:
            # Python fallback safe implementation (vectorized loop)
            # h shape: (B, E, N)
            device = A.device
            h = torch.zeros(Bsz, E, N, dtype=orig_dtype, device=device)
            outputs = []
            for t in range(M):
                A_t = A[:, t]  # (B, E, N)
                B_t = B[:, t]  # (B, E, N)
                C_t = C[:, t]  # (B, E, N)
                x_t = X[:, t].unsqueeze(-1)  # (B, E, 1)
                h = A_t * h + B_t * x_t     # elementwise multiply / broadcast
                y_t = (h * C_t).sum(dim=-1) # (B, E)
                outputs.append(y_t)
            return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, M, D)
        returns: (B, M, D)
        """
        B, M, D = x.shape

        # normalize + project
        x_norm = self.norm(x)
        x_proj = self.to_x(x_norm)   # (B, M, E)
        z = self.to_z(x_norm)        # (B, M, E)

        # ---- forward branch ----
        # conv expects (B, E, M)
        x_fwd = F.silu(self.conv_fwd(x_proj.transpose(1, 2))).transpose(1, 2)  # (B, M, E)

        B_f = self.B_fwd(x_fwd).view(B, M, self.expand, self.ssm_dim)   # (B, M, E, N)
        C_f = self.C_fwd(x_fwd).view(B, M, self.expand, self.ssm_dim)   # (B, M, E, N)
        # Δ_fwd: (B, M, E)
        delta_f_pre = self.D_fwd(x_fwd) + self.DeltaBias_fwd.view(1, 1, -1)
        delta_f = F.softplus(delta_f_pre)

        # Ao_fwd = Δ_fwd * ParameterA_fwd  -> shape (B, M, E, N)
        # keep ParameterA in fp32 and cast to device
        ParamA_f = self.ParameterA_fwd.float().to(x_proj.device)
        Ao_f = delta_f.unsqueeze(-1) * ParamA_f.view(1, 1, self.expand, self.ssm_dim)

        # B_full_fwd = Δ_fwd * B_f (per Algo 1)
        B_full_f = delta_f.unsqueeze(-1) * B_f  # (B, M, E, N)

        # ---- backward branch ----
        x_bwd = F.silu(self.conv_bwd(x_proj.transpose(1, 2))).transpose(1, 2)  # (B, M, E)
        B_b = self.B_bwd(x_bwd).view(B, M, self.expand, self.ssm_dim)
        C_b = self.C_bwd(x_bwd).view(B, M, self.expand, self.ssm_dim)
        delta_b_pre = self.D_bwd(x_bwd) + self.DeltaBias_bwd.view(1, 1, -1)
        delta_b = F.softplus(delta_b_pre)
        ParamA_b = self.ParameterA_bwd.float().to(x_proj.device)
        Ao_b = delta_b.unsqueeze(-1) * ParamA_b.view(1, 1, self.expand, self.ssm_dim)
        B_full_b = delta_b.unsqueeze(-1) * B_b

        # ---- selective scan (forward) ----
        # y_fwd: (B, M, E)
        y_fwd = self._selective_scan(Ao_f, B_full_f, C_f, x_fwd)

        # ---- selective scan (backward) ----
        # process reversed sequence then flip back to get backward outputs
        x_bwd_rev = x_bwd.flip(dims=[1])                       # (B, M, E)
        Ao_b_rev = Ao_b.flip(dims=[1])                         # (B, M, E, N)
        B_full_b_rev = B_full_b.flip(dims=[1])
        C_b_rev = C_b.flip(dims=[1])

        y_bwd_rev = self._selective_scan(Ao_b_rev, B_full_b_rev, C_b_rev, x_bwd_rev)  # (B, M, E)
        y_bwd = y_bwd_rev.flip(dims=[1])  # (B, M, E)

        # ---- gate and combine ----
        y = (y_fwd + y_bwd) * F.silu(z)   # (B, M, E)
        out = self.to_out(y)             # (B, M, D)

        return x + out


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
