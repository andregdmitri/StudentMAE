# train/retfound_eval.py

import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from torchvision import transforms

from config.constants import *
from models.retfound import RETFoundClassifier
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule


# ------------------------------------------------------------------------------
# TRAINING WRAPPER
# ------------------------------------------------------------------------------

class RETFoundTrainer(pl.LightningModule):
    """
    RETFound + new classification head trained on IDRiD.
    """

    def __init__(self, backbone, lr, class_weights=None, finetune=False):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        # freeze backbone unless user wants full finetuning
        if not finetune:
            for p in backbone.parameters():
                p.requires_grad = False

        self.backbone = backbone.eval() if not finetune else backbone

        # Classification head (5 classes for IDRiD)
        self.head = nn.Sequential(
            nn.Linear(backbone.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES)
        )

        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.acc.update(preds, y)
        self.f1.update(preds, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/acc", self.acc.compute())
        self.log("train/f1", self.f1.compute())
        self.acc.reset()
        self.f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.acc.update(preds, y)
        self.f1.update(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val/acc", self.acc.compute())
        self.log("val/f1", self.f1.compute())
        self.acc.reset()
        self.f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ------------------------------------------------------------------------------
# EVALUATION WRAPPER (unchanged)
# ------------------------------------------------------------------------------

class RETFoundEvalWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.acc.update(preds, y)
        self.f1.update(preds, y)
        try:
            self.auroc.update(probs, y)
            self.aupr.update(probs, y)
        except:
            pass

    def on_validation_epoch_end(self):
        print("\n=== RETFound Evaluation ===")
        print("ACC:", self.acc.compute())
        print("F1:", self.f1.compute())
        print("AUROC:", self.auroc.compute())
        print("AUPR:", self.aupr.compute())

        self.acc.reset()
        self.f1.reset()
        self.auroc.reset()
        self.aupr.reset()


# ------------------------------------------------------------------------------
# BUILD RETFOUND
# ------------------------------------------------------------------------------

def build_retfound(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")

    return RETFoundClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=checkpoint_path
    ).eval()


# ------------------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------------------

def run_retfound_training(args):
    print("\n=== RETFound Training on IDRiD ===")

    # 1. Build backbone
    backbone = build_retfound(args.checkpoint)

    # 2. Class weights
    csv_path = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    class_weights = compute_idrid_class_weights(csv_path, NUM_CLASSES)

    # 3. Create trainer module
    model = RETFoundTrainer(
        backbone=backbone,
        lr=args.lr,
        class_weights=class_weights,
        finetune=args.finetune
    )

    # 4. Data
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup()

    # 5. Callbacks
    ckpt_cb = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="retfound_idrid",
    )

    early_cb = EarlyStopping(
        monitor="val/f1",
        patience=30,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="retfound_idrid_train"),
        callbacks=[ckpt_cb, early_cb],
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    trainer.fit(model, datamodule=dm)

    print("\nSaved:", ckpt_cb.best_model_path)
    return ckpt_cb.best_model_path


# ------------------------------------------------------------------------------
# EVALUATION ENTRY
# ------------------------------------------------------------------------------

def run_retfound_evaluation(dataset, checkpoint_path=None):
    print("\n=== RETFound Evaluation Mode ===")

    model = build_retfound(checkpoint_path)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    if dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup(stage="validate")

    wrapper = RETFoundEvalWrapper(model)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="retfound_original_eval"),
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )
    trainer.validate(wrapper, datamodule=dm)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", type=str, required=True,
                   choices=["train", "eval"])

    p.add_argument("--dataset", type=str, default="idrid",
                   choices=["idrid", "aptos"])

    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--finetune", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        run_retfound_training(args)

    elif args.mode == "eval":
        run_retfound_evaluation(dataset=args.dataset, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
