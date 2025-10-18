import torch
import torch.nn as nn
from mamba_ssm import Mamba

# Import the BaseClassifier from your file
from .base_model import BaseClassifier

class VisualMamba(BaseClassifier):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 24,
        learning_rate: float = 0.001,
        mask_ratio: float = 0.0,
        **mamba_kwargs,
    ):
        # The BaseClassifier's `input_dim` isn't directly used by this new model's
        # forward pass, but we pass `embed_dim` to satisfy the parent constructor.
        super().__init__(input_dim=embed_dim, num_classes=num_classes, learning_rate=learning_rate)
        
        # This is a PyTorch Lightning convention to save hyperparameters.
        self.save_hyperparameters()

        # 1. Patch Embedding Layer
        # This layer converts an image into a sequence of flattened patch embeddings.
        # For example, a 224x224 image with 16x16 patches becomes 14x14 = 196 patches.
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.mask_ratio = mask_ratio
        self.mask_token = None # incomplete

        # 2. Mamba Backbone
        # A sequence of Mamba blocks to process the patch embeddings.
        self.backbone = nn.Sequential(
            *[
                Mamba(d_model=embed_dim, **mamba_kwargs)
                for _ in range(depth)
            ]
        )

        # 3. Classification Head
        # A normalization layer and a linear layer to produce the final logits.
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def apply_mask(self, x: torch.Tensor):
      B, N, D = x.shape
      num_mask = int(N * self.mask_ratio)

      noise = torch.rand(B, N, device=x.device)
      ids_shuffle = torch.argsort(noise, dim=1)
      ids_restore = torch.argsort(ids_shuffle, dim=1)

      mask = torch.ones(B, N, device=x.device)
      mask[:, :num_mask] = 0
      mask = torch.gather(mask, dim=1, index=ids_restore)

      if self.mask_token is not None:
        mask_tokens = self.mask_token.expand(B, N, D)
        x = x * mask.unsqueeze(-1) + (1 - mask).unsqueeze(-1) * mask_tokens
      else:
        x = x * mask.unsqueeze(-1)

      return x, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.patch_embed(x)
        
        # Flatten the spatial dimensions and permute for sequence processing
        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D) where N is the number of patches
        x = x.flatten(2).transpose(1, 2)

        mask = None
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
        x = self.head(x)
        
        return x
