import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import models_vit  # from RETFound repo
from util.pos_embed import interpolate_pos_embed
from base_model import BaseClassifier

class RETFoundClassifier(BaseClassifier):
    def __init__(self, output_dim, checkpoint_path, learning_rate=1e-4, drop_path_rate=0.2):
        # We don't really need input_dim here, so just pass None
        super().__init__(input_dim=None, output_dim=output_dim, learning_rate=learning_rate)

        # RETFound has this ViT backbone
        self.model = models_vit.__dict__['vit_large_patch16'](
            num_classes=output_dim,
            drop_path_rate=drop_path_rate,
            global_pool=True,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()

        # Remove incompatible classification head weights
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Interpolate position embedding for different image sizes if needed
        interpolate_pos_embed(self.model, checkpoint_model)

        # Load pretrained encoder weights
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys after loading checkpoint:", msg.missing_keys)

        # Re-initialize head
        trunc_normal_(self.model.head.weight, std=2e-5)

    def forward(self, x):
        return self.model(x)