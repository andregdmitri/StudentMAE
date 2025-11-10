import torch
import torch.nn as nn
from mamba_ssm import Mamba
from config.constants import *
import torch.nn.functional as F

# Import the BaseClassifier from your file
from .base_model import BaseClassifier

class VisualMamba(BaseClassifier):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 192,     # D
        depth: int = 24,          # L
        ssm_dim: int = 16,        # N
        expand_dim: int = 2,      # E
        learning_rate: float = LEARNING_RATE,
        variant: str = "tiny",    # "tiny" or "small"
        mask_ratio: float = 0.75,
        **mamba_kwargs,
    ):
        # The BaseClassifier's `input_dim` isn't directly used by this new model's
        # forward pass, but we pass `embed_dim` to satisfy the parent constructor.
        super().__init__(input_dim=embed_dim, num_classes=num_classes, learning_rate=learning_rate)
        
        # This is a PyTorch Lightning convention to save hyperparameters.
        self.save_hyperparameters()

        self.mask_ratio = mask_ratio

        # 1. Patch Embedding Layer
        # This layer converts an image into a sequence of flattened patch embeddings.
        # For example, a 224x224 image with 16x16 patches becomes 14x14 = 196 patches.
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
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
        self.head = nn.Linear(embed_dim, num_classes)
        if mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None

    def apply_mask(self, x: torch.Tensor):
        B, N, D = x.shape
        num_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, num_mask:]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1,-1,D))

        # create binary mask (1=keep, 0=masked)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_mask] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if self.mask_token is not None:
            # pad masked tokens
            mask_tokens = self.mask_token.expand(B, num_mask, D)
            x = torch.cat([x, mask_tokens], dim=1)

        return x, mask

    def forward(self, x):
        feats = self.forward_features(x)        # (B,D)
        logits = self.head(feats)               # (B,5)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.patch_embed(x)
        
        # Flatten the spatial dimensions and permute for sequence processing
        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D) where N is the number of patches
        x = x.flatten(2).transpose(1, 2)

        # mask = None
        if self.mask_ratio > 0.0:
            x, mask = self.apply_mask(x)  # (B, N, D), (B, N)
        
        # Process the sequence through the Mamba backbone
        # (B, N, D) -> (B, N, D)
        x = self.backbone(x)
        
        # Aggregate the sequence information via global average pooling
        # (B, N, D) -> (B, D)
        x = x.mean(dim=1)
        
        # Pass through the classification head to get logits
        # (B, D) -> (B, D) -> (B, num_classes)
        x = self.norm(x)
        # x = self.head(x)
        
        return x