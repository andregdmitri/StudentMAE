import torch
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger

from config.constants import *
from train.train_retfound import RETFoundTask
from dataloader.idrid import IDRiDModule
from dataloader.aptos import APTOSModule
from utils.flops import compute_flops
from eval.shared_eval import EvalWrapper

def run_eval_retfound(args):
    print(f"\n=== RETFOUND EVALUATION: {args.load_model} ===")

    # 1. Load the Task (handles Backbone + Head automatically)
    model = RETFoundTask.load_from_checkpoint(args.load_model, strict=False)
    
    # 2. Setup Data
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup(stage="validate")

    # 3. Evaluate
    wrapper = EvalWrapper(model)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        logger=WandbLogger(project="retfound_eval")
    )
    
    trainer.validate(wrapper, dm)

    # 4. Complexity
    flops, _ = compute_flops(model, IMG_SIZE)
    print(f"Total Complexity: {flops/1e9:.3f} GFLOPs")