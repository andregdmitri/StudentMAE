# train/head.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision

from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule
from torchvision import transforms
import os


class HeadTrainer(pl.LightningModule):
    """
    Phase II trainer:
    - Loads frozen backbone
    - Trains classification head only
    """

    def __init__(self, backbone, lr):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.eval()

        # trainable head
        self.head = nn.Linear(backbone.embed_dim, NUM_CLASSES)

        self.loss_fn = nn.CrossEntropyLoss()

        # -------- Metrics --------
        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
        return self.head(feats)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # CORRECT
        self.acc.update(preds, y)
        self.f1.update(preds, y)

        self.auroc.update(probs, y)
        self.aupr.update(probs, y)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        acc = self.acc.compute()
        f1 = self.f1.compute()

        # multiclass AUROC/AUPR can throw if class missing → catch it
        try:
            auroc = self.auroc.compute()
        except:
            auroc = torch.tensor(float("nan"))

        try:
            aupr = self.aupr.compute()
        except:
            aupr = torch.tensor(float("nan"))

        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1)
        self.log("val/auroc", auroc)
        self.log("val/aupr", aupr)

        # reset
        self.acc.reset()
        self.f1.reset()
        self.auroc.reset()
        self.aupr.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )


def run_head_training(args):
    pl.seed_everything(42)
    logger = WandbLogger(project="vmamba_head_training")

    print("=== PHASE II: HEAD TRAINING ===")

    # ------------------------------
    # 1. Load frozen backbone
    # ------------------------------
    if args.load_backbone is None:
        raise ValueError("--load_backbone required for head training")

    print(f"Loading backbone: {args.load_backbone}")

    backbone = VisualMamba(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        learning_rate=0.0,
        mask_ratio=0.0,
    )

    ckpt = torch.load(args.load_backbone, map_location="cpu")

    if "backbone" in ckpt:
        backbone.load_state_dict(ckpt["backbone"])
    else:
        backbone.load_state_dict(ckpt)

    # ------------------------------
    # 2. Build HeadTrainer
    # ------------------------------
    lr = args.head_lr if args.head_lr else HEAD_LR
    model = HeadTrainer(backbone, lr)

    # ------------------------------
    # 3. Data
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

    # ------------------------------
    # 4. Trainer
    # ------------------------------
    epochs = args.head_epochs if args.head_epochs else HEAD_EPOCHS

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
    )

    # ------------------------------
    # 5. Train
    # ------------------------------
    trainer.fit(model, datamodule=dm)

    # ------------------------------
    # 6. Save final backbone + head
    # ------------------------------
    save_path = os.path.join(CHECKPOINT_DIR, "vmamba_final_head.pth")
    torch.save({
        "backbone": backbone.state_dict(),
        "head": model.head.state_dict(),
    }, save_path)

    print(f"[✓] Saved final model → {save_path}")