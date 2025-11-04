import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from config.constants import *
from . import models_vit  # from RETFound repo
from utils.pos_embed import interpolate_pos_embed
from .base_model import BaseClassifier

class RETFoundClassifier(BaseClassifier):
    def __init__(self, num_classes=NUM_CLASSES, checkpoint_path=os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth"), learning_rate=LEARNING_RATE, drop_path_rate=0.2):
        # We don't really need input_dim here, so just pass None
        super().__init__(input_dim=None, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

        # RETFound has this ViT backbone
        self.model = models_vit.__dict__['vit_large_patch16'](
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=True,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
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
    
    def forward_features(self, x):
        return self.model.forward_features(x)
