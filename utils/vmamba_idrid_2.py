# train_vmamba_idrid.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from torchvision import transforms

from config.constants import *
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from models.vmamba_backbone import VisualMamba
import os


class VmambaSupervised(pl.LightningModule):
    """
    Direct supervised training:
        VisualMamba (trainable) + classification head (trainable)
    """

    def __init__(self, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        # -----------------------------
        # Backbone (trainable)
        # -----------------------------
        self.backbone = VisualMamba(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANS,
            embed_dim=VMAMBA_EMBED_DIM,
            depth=VMAMBA_DEPTH,
            learning_rate=lr,
            mask_ratio=0.0,            # No masking
            use_cls_token=False,       # same as in your distillation script
        )

        # -----------------------------
        # Classification head
        # -----------------------------
        self.head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 512),
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

    # -----------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)
        try:
            self.train_auroc.update(probs, y)
            self.train_aupr.update(probs, y)
        except:
            pass

        self.log("train/loss", loss, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), prog_bar=True)
        self.log("train/f1", self.train_f1.compute())
        try:
            self.log("train/auroc", self.train_auroc.compute())
            self.log("train/aupr", self.train_aupr.compute())
        except:
            pass

        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()
        self.train_aupr.reset()

    # -----------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------
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

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute())

        try:
            self.log("val/auroc", self.val_auroc.compute())
            self.log("val/aupr", self.val_aupr.compute())
        except:
            pass

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()

    # -----------------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )


# ===================================================================
#                        RUN SUPERVISED TRAINING
# ===================================================================

def run_vmamba_supervised_training():
    pl.seed_everything(42)

    logger = WandbLogger(project="vmamba_supervised")

    # class weights
    csv_path = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    class_weights = compute_idrid_class_weights(csv_path, NUM_CLASSES)

    model = VmambaSupervised(
        lr=HEAD_LR,            # reuse your constant
        class_weights=class_weights,
    )

    # Dataset
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dm = IDRiDModule(
        root=IDRID_PATH,
        transform=tfm,
        batch_size=BATCH_SIZE,
    )
    dm.setup()

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="vmamba_supervised_best",
    )

    early_stop = EarlyStopping(
        monitor="val/f1",
        patience=30,
        mode="max"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=HEAD_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint, early_stop],
    )

    trainer.fit(model, datamodule=dm)

    # -------------------------------------------------------------
    # Save final clean backbone + head
    # -------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, "vmamba_supervised_final.pth")

    ckpt = torch.load(checkpoint.best_model_path, map_location="cpu")
    state = ckpt["state_dict"]

    backbone_state = {}
    head_state = {}

    for k, v in state.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
        elif k.startswith("head."):
            head_state[k[len("head."):]] = v

    torch.save({"backbone": backbone_state, "head": head_state}, save_path)

    print(f"\n[✓] Saved supervised Vmamba model → {save_path}\n")


if __name__ == "__main__":
    run_vmamba_supervised_training()
