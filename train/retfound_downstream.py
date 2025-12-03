# train/retfound_downstream.py

import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.flops import compute_flops
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision
from torchvision import transforms

from config.constants import *
from models.retfound import RETFoundClassifier
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule


# -----------------------------------------------------------
#  Helpers
# -----------------------------------------------------------

def load_retfound_backbone(path=None):
    """
    Loads RETFound full model and returns ONLY the ViT backbone (frozen).
    Ensures the backbone.forward_features(x) returns (B, D) pooled features.
    """
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")

    full = RETFoundClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=path,
    ).eval()

    vit = full.model  # timm ViT-Large
    embed_dim = vit.head.weight.shape[1]  # correct and reliable
    return vit, embed_dim


def get_feats(backbone, x):
    """RETFound ViT-L returns pooled features (B, D)."""
    with torch.no_grad():
        feats = backbone.forward_features(x)  # already pooled
    return feats


# -----------------------------------------------------------
#  Lightning Module - TRUE LINEAR PROBING
# -----------------------------------------------------------

class RETFoundLinearProbe(pl.LightningModule):

    def __init__(self, backbone, embed_dim, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.eval()

        # ✔ TRUE linear probe (one layer)
        self.head = nn.Linear(embed_dim, NUM_CLASSES)

        # Loss
        if class_weights is not None:
            self.register_buffer("cw", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.cw)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # -------- Training Metrics --------
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.train_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.train_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.train_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

        # -------- Validation Metrics --------
        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.val_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        feats = get_feats(self.backbone, x)
        return self.head(feats)

    # ----------------------- TRAIN -----------------------
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # --- Metrics ---
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        try:
            self.train_auroc.update(probs, y)
            self.train_aupr.update(probs, y)
        except:
            pass  # AUROC/AUPR crashes when batch missing classes

        # Log loss now; metrics logged on epoch end
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))

        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        f1 = self.train_f1.compute()

        try:
            auroc = self.train_auroc.compute()
        except:
            auroc = torch.tensor(float("nan"))

        try:
            aupr = self.train_aupr.compute()
        except:
            aupr = torch.tensor(float("nan"))

        self.log("train/acc", acc, prog_bar=True)
        self.log("train/f1", f1)
        self.log("train/auroc", auroc)
        self.log("train/aupr", aupr)

        # Reset
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()
        self.train_aupr.reset()

    # ----------------------- VAL -----------------------
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        try:
            self.val_auroc.update(probs, y)
            self.val_aupr.update(probs, y)
        except:
            pass

        self.log("val/loss", loss, on_step=False, prog_bar=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()

        try:
            auroc = self.val_auroc.compute()
        except:
            auroc = torch.tensor(float("nan"))

        try:
            aupr = self.val_aupr.compute()
        except:
            aupr = torch.tensor(float("nan"))

        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1)
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/aupr", aupr)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()

    # ----------------------- OPTIM -----------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


# -----------------------------------------------------------
#  TRAIN LOOP
# -----------------------------------------------------------

def run_train(args):

    pl.seed_everything(42)

    print("\n=== RETFound LINEAR PROBE TRAINING ===")

    # 1. Load backbone
    backbone, embed_dim = load_retfound_backbone(args.checkpoint)

    # 2. Class weights
    csv = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    class_weights = compute_idrid_class_weights(csv, NUM_CLASSES)

    # 3. Model
    model = RETFoundLinearProbe(
        backbone=backbone,
        embed_dim=embed_dim,
        lr=args.lr,
        class_weights=class_weights,
    )

    # 4. Strong augmentations (closer to RETFound paper)
    tfm = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
    ])

    dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup(stage="full")

    ckpt_cb = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="retfound_linearprobe_best",
    )
    early = EarlyStopping(monitor="val/f1", patience=40, mode="max")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="retfound_linearprobe"),
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[ckpt_cb, early],
        log_every_n_steps=4,
    )

    trainer.fit(model, dm)

    print(f"\n[✓] Best checkpoint: {ckpt_cb.best_model_path}")


# -----------------------------------------------------------
#  EVAL LOOP
# -----------------------------------------------------------

class EvalWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

        # -------- Evaluation Metrics --------
        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.val_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        try:
            self.val_auroc.update(probs, y)
            self.val_aupr.update(probs, y)
        except:
            pass  # batch missing some classes → safe skip

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()

        try:
            auroc = self.val_auroc.compute()
        except:
            auroc = torch.tensor(float("nan"))

        try:
            aupr = self.val_aupr.compute()
        except:
            aupr = torch.tensor(float("nan"))

        print("\n=== EVAL RESULTS ===")
        print("ACC :", acc)
        print("F1  :", f1)
        print("AUROC:", auroc)
        print("AUPR :", aupr)

        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1", f1)
        self.log("val/auroc", auroc)
        self.log("val/aupr", aupr)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()

def run_eval(dataset: str, ckpt: str):
    print("\n=== LINEAR PROBE EVALUATION ===")

    # Backbone must match RETFound pretraining
    backbone, embed_dim = load_retfound_backbone()

    # Load Lightning module (ignore mismatched keys)
    model = RETFoundLinearProbe.load_from_checkpoint(
        ckpt,
        backbone=backbone,
        embed_dim=embed_dim,
        lr=1e-6,
        strict=False,
    ).eval()

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    if dataset == "idrid":
        dm = IDRiDModule(
            root=IDRID_PATH,
            transform=tfm,
            batch_size=BATCH_SIZE
        )
        dm.setup()
    else:
        dm = APTOSModule(APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup(stage="full")

    wrapper = EvalWrapper(model)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=WandbLogger(project="retfound_linearprobe_eval"),
    )

    trainer.validate(wrapper, dm)
        # Compute FLOPs
    total_flops, total_params = compute_flops(model, IMG_SIZE)

    print("\n=== MODEL COMPLEXITY ===")
    print(f"FLOPs:   {total_flops/1e9:.2f} GFLOPs")


# -----------------------------------------------------------
#  CLI
# -----------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--dataset", choices=["idrid", "aptos"], default="idrid")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args.dataset, args.checkpoint)


if __name__ == "__main__":
    main()
