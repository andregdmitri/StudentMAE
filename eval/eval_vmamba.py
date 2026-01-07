import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule
from dataloader.aptos import APTOSModule
from utils.flops import compute_flops
from eval.shared_eval import EvalWrapper

def run_evaluation(args):
    print(f"\n=== VMAMBA EVALUATION: {args.load_model} ===")

    # 1. Reconstruct Model Structure
    backbone = VisualMamba(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM, depth=VMAMBA_DEPTH, 
        learning_rate=0.0, mask_ratio=0.0, use_cls_token=False
    )
    
    head = nn.Sequential(
        nn.Linear(backbone.embed_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(128, NUM_CLASSES)
    )

    # 2. Load State Dict
    ckpt = torch.load(args.load_model, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    
    # Combine into a single inference module
    class VMambaInference(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h
        def forward(self, x):
            return self.head(self.backbone.forward_features(x))

    model = VMambaInference(backbone, head)

    # 3. Data & Trainer
    tfm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE) if args.dataset == "idrid" \
         else APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup(stage="validate")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="vmamba_eval"),
        precision="16-mixed"
    )

    # 4. Run
    wrapper = EvalWrapper(model)
    trainer.validate(wrapper, dm)

    flops, _ = compute_flops(model, IMG_SIZE)
    print(f"Total Complexity: {flops/1e9:.3f} GFLOPs")