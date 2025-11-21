# train/eval.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision

from config.constants import *
from models.vmamba_backbone import VisualMamba
from torchvision import transforms
from dataloader.idrid import IDRiDModule


class EvalWrapper(pl.LightningModule):
    """
    Wraps backbone + head to compute evaluation metrics.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone.eval()
        self.head = head.eval()

        # metrics
        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            logits = self.head(feats)
        return logits

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # CORRECT
        self.acc.update(preds, y)
        self.f1.update(preds, y)

        self.auroc.update(probs, y)
        self.aupr.update(probs, y)

    def on_validation_epoch_end(self):
        acc = self.acc.compute()
        f1 = self.f1.compute()

        try:
            auroc = self.auroc.compute()
        except:
            auroc = torch.tensor(float("nan"))

        try:
            aupr = self.aupr.compute()
        except:
            aupr = torch.tensor(float("nan"))

        self.log("eval/acc", acc)
        self.log("eval/f1", f1)
        self.log("eval/auroc", auroc)
        self.log("eval/aupr", aupr)

        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUROC:    {auroc:.4f}")
        print(f"AUPR:     {aupr:.4f}")

        self.acc.reset()
        self.f1.reset()
        self.auroc.reset()
        self.aupr.reset()


def run_evaluation(args):
    pl.seed_everything(42)

    if args.load_model is None:
        raise ValueError("--load_model is required for evaluation")

    print(f"\n=== Loading model checkpoint ===\n{args.load_model}")

    ckpt = torch.load(args.load_model, map_location="cpu")

    # ------------------------------
    # Build backbone
    # ------------------------------
    backbone = VisualMamba(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        learning_rate=0.0,
        mask_ratio=0.0,
    )

    if "backbone" in ckpt:
        backbone.load_state_dict(ckpt["backbone"])
    else:
        backbone.load_state_dict(ckpt, strict=False)

    # ------------------------------
    # Build classifier head
    # ------------------------------
    head = nn.Linear(backbone.embed_dim, NUM_CLASSES)

    if "head" in ckpt:
        head.load_state_dict(ckpt["head"])
    else:
        print("WARNING: No head found — random head!")
    head.eval()

    # ------------------------------
    # Data
    # ------------------------------
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dm = IDRiDModule(
        root=IDRID_PATH,
        transform=tfm,
        batch_size=BATCH_SIZE
    )
    dm.setup()

    logger = WandbLogger(project="vmamba_eval")

    model = EvalWrapper(backbone, head)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    print("\n=== Running Evaluation ===")
    trainer.validate(model, datamodule=dm)
    print("\n=== Done ===\n")