# train_vmamba_idrid.py

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from torchvision import transforms

from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights


class VmambaClassifier(pl.LightningModule):
    """
    Full supervised training of Vmamba on IDRiD (no distillation).
    """

    def __init__(self, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        # Backbone (trainable)
        self.backbone = VisualMamba(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANS,
            embed_dim=VMAMBA_EMBED_DIM,
            depth=VMAMBA_DEPTH,
            learning_rate=lr,
            mask_ratio=0.0,         # disable MAE masking
        )

        # Classification head
        self.head = nn.Linear(self.backbone.embed_dim, NUM_CLASSES)

        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.train_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.train_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.train_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.val_aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        return self.head(feats)

    # ----------------------------
    # TRAIN
    # ----------------------------
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)
        self.train_auroc.update(probs, y)
        self.train_aupr.update(probs, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ----------------------------
    # VALIDATION
    # ----------------------------
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_auroc.update(probs, y)
        self.val_aupr.update(probs, y)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute())
        self.log("val/auroc", self.val_auroc.compute())
        self.log("val/aupr", self.val_aupr.compute())

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()

    # ----------------------------
    # OPTIMIZER
    # ----------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )

    def predict_step(self, batch, batch_idx):
        x, y, paths = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return {
            "paths": paths,
            "labels": y.cpu(),
            "preds": preds.cpu(),
            "probs": probs.cpu(),
        }
    
# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------

def train_vmamba_idrid():
    pl.seed_everything(42)

    # Class weights
    csv_path = os.path.join(
        IDRID_PATH,
        "2. Groundtruths",
        "a. IDRiD_Disease Grading_Training Labels.csv"
    )
    class_weights = compute_idrid_class_weights(csv_path, NUM_CLASSES)

    # Transform
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Data
    dm = IDRiDModule(
        root=IDRID_PATH,
        transform=tfm,
        batch_size=BATCH_SIZE
    )
    dm.setup()

    # Logger
    logger = WandbLogger(project="vmamba_full_supervised")

    # Model
    model = VmambaClassifier(
        lr=HEAD_LR,
        class_weights=class_weights
    )

    # Callbacks
    ckpt = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="vmamba_supervised_best",
    )

    early = EarlyStopping(
        monitor="val/loss",
        patience=PATIENCE,
        mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=HEAD_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[ckpt, early],
        log_every_n_steps=5,
    )

    print("\n=== Training Visual Mamba (NO DISTILLATION) ===\n")
    trainer.fit(model, datamodule=dm)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "vmamba_supervised_final.pth")
    torch.save(torch.load(ckpt.best_model_path), final_path)

    print(f"\n[✓] Saved model → {final_path}\n")
    
    def predict_step(self, batch, batch_idx):
        x, y, paths = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {
            "paths": paths,
            "labels": y.cpu(),
            "preds": preds.cpu(),
            "probs": probs.cpu(),
        }


if __name__ == "__main__":
    train_vmamba_idrid()
