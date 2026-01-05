# train/train_retfound.py

import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import *
from models.retfound import RETFoundClassifier
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule


# ------------------------------------------------------------
# Backbone loading
# ------------------------------------------------------------

def load_retfound_backbone(path=None):
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")

    print(f"[INFO] Loading RETFound backbone from {path}")

    model = RETFoundClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=path
    ).eval()

    vit = model.model
    embed_dim = vit.head.weight.shape[1]

    return vit, embed_dim


def get_feats(backbone, x):
    with torch.no_grad():
        feats = backbone.forward_features(x)
    return feats


# ------------------------------------------------------------
# Linear Probe Lightning Module
# ------------------------------------------------------------

class RETFoundLinearProbe(pl.LightningModule):
    def __init__(self, backbone, embed_dim, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.eval()

        self.head = torch.nn.Linear(embed_dim, NUM_CLASSES)

        if class_weights is not None:
            self.register_buffer("cw", class_weights)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.cw)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        feats = get_feats(self.backbone, x)
        return self.head(feats)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=self.hparams.lr)


# ------------------------------------------------------------
# Fine-tuning Lightning Module
# ------------------------------------------------------------

class RETFoundFineTune(pl.LightningModule):
    def __init__(self, model, lr, class_weights=None):
        super().__init__()
        self.model = model
        self.lr = lr

        if class_weights is not None:
            self.register_buffer("cw", class_weights)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.cw)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)


# ------------------------------------------------------------
# Unified Train Entry
# ------------------------------------------------------------

def run_train_retfound(args):
    pl.seed_everything(42)
    print("\n=== RETFound Training Mode:", args.retfound_mode, "===\n")

    # Backbone
    backbone, embed_dim = load_retfound_backbone(args.checkpoint)

    # Class weights (IDRiD only)
    csv = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    class_weights = compute_idrid_class_weights(csv, NUM_CLASSES)

    # Dataset
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    if args.dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup()
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup(stage="train")

    # Choose model
    if args.retfound_mode == "linear":
        model = RETFoundLinearProbe(
            backbone=backbone,
            embed_dim=embed_dim,
            lr=args.lr,
            class_weights=class_weights,
        )
    elif args.retfound_mode == "finetune":
        full = RETFoundClassifier(num_classes=NUM_CLASSES)
        model = RETFoundFineTune(
            model=full,
            lr=args.lr,
            class_weights=class_weights,
        )
    else:
        raise ValueError("Invalid mode")

    # Callbacks
    ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val/loss", patience=40, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early],
        logger=WandbLogger(project="retfound_training"),
    )

    trainer.fit(model, dm)
