# train.retfound_finetune.py

import os
import argparse
from typing import Optional
from optimizers.optmizer import warmup_cosine_optimizer

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision
from torchvision import transforms

from config.constants import *
from models.retfound import RETFoundClassifier
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule
from utils.flops import compute_flops


# -----------------------------------------------------------
#  HELPERS
# -----------------------------------------------------------

def load_retfound_backbone(path=None):
    """ Loads RETFound and returns backbone + embed dim. """
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")

    full = RETFoundClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=path,
    ).eval()

    vit = full.model
    embed_dim = vit.head.weight.shape[1]
    return vit, embed_dim


def get_feats(backbone, x):
    return backbone.forward_features(x)


# -----------------------------------------------------------
#  FINE-TUNING MODULE (END-TO-END)
# -----------------------------------------------------------

class RETFoundFineTune(pl.LightningModule):

    def __init__(self, backbone, embed_dim, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        # **UNFREEZE BACKBONE → FULL FINETUNING**
        for p in backbone.parameters():
            p.requires_grad = True

        self.backbone = backbone

        # Classifier head
        self.head = nn.Linear(embed_dim, NUM_CLASSES)

        # Loss
        if class_weights is not None:
            self.register_buffer("cw", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.cw)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

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

    # ---------------- TRAIN ----------------
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

    # ---------------- VALIDATION ----------------
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

    # ---------------- OPTIM ----------------
    def configure_optimizers(self):
        params = list(self.head.parameters())
        if not FREEZE_BACKBONE:
            params += list(self.backbone.parameters())

        optimizer, scheduler = warmup_cosine_optimizer(
            parameters=params,
            max_epochs=self.trainer.max_epochs,
            lr=self.hparams.lr,
            warmup_epochs=WARMUP_EPOCHS,
            final_lr=FINAL_LR,
            weight_decay=WEIGHT_DECAY
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


# -----------------------------------------------------------
#  TRAIN LOOP (TRAIN ON IDRID + VALIDATE ON IDRID + APTOS)
# -----------------------------------------------------------

def run_train(args):
    pl.seed_everything(42)

    print("\n=== RETFound FULL FINETUNING ===")

    # 1. Backbone
    backbone, embed_dim = load_retfound_backbone(args.checkpoint)

    # 2. Class weights
    csv = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    class_weights = compute_idrid_class_weights(csv, NUM_CLASSES)

    # 3. Model
    model = RETFoundFineTune(
        backbone=backbone,
        embed_dim=embed_dim,
        lr=LR,
        class_weights=class_weights,
    )

    # 4. Data transforms
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Training on IDRiD
    dm_idrid = IDRiDModule(root=IDRID_PATH, transform=tf, batch_size=BATCH_SIZE)
    dm_idrid.setup()

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="retfound_finetune_best",
    )
    early = EarlyStopping(monitor="val/f1", patience=50, mode="max")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=WandbLogger(project="retfound_finetune_dualval"),
        callbacks=[ckpt_cb, early],
        log_every_n_steps=4,
    )

    trainer.fit(
        model,
        train_dataloaders=dm_idrid.train_dataloader(),
        val_dataloaders=dm_idrid.val_dataloader(),
    )

    print(f"\n[✓] Best checkpoint: {ckpt_cb.best_model_path}")


# -----------------------------------------------------------
#  EVAL (OPTIONAL)
# -----------------------------------------------------------

def run_eval(dataset: str, ckpt: str):
    print("\n=== EVALUATION ===")

    backbone, embed_dim = load_retfound_backbone()

    model = RETFoundFineTune.load_from_checkpoint(
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
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup()
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup(stage="validate")

    wrapper = model

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        logger=WandbLogger(project="retfound_finetune_eval"),
    )

    trainer.validate(wrapper, dm)

    total_flops, _ = compute_flops(model, IMG_SIZE)
    print(f"\nFLOPs: {total_flops/1e9:.2f} GFLOPs")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--dataset", choices=["idrid", "aptos"], default="idrid")
    p.add_argument("--checkpoint", type=str, default=None)
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

# python -m train.retfound_finetune --mode train --dataset idrid --checkpoint path/to/checkpoint.pth