# train/retfound_downstream.py
import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from torchvision import transforms

from config.constants import *
from models.retfound import RETFoundClassifier
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule


# -----------------------
# Utility helpers
# -----------------------
def resolve_backbone(backbone_or_fullmodel):
    """
    Accept either:
     - a RETFoundClassifier (which has .model)
     - or a ViT backbone directly
    Returns the backbone module (ViT-like) and embed_dim (int).
    """
    # If user passed the full RETFoundClassifier, extract .model
    if hasattr(backbone_or_fullmodel, "model"):
        backbone = backbone_or_fullmodel.model
    else:
        backbone = backbone_or_fullmodel

    # Try common attributes for embed dim
    embed_dim = None
    if hasattr(backbone, "embed_dim"):
        embed_dim = getattr(backbone, "embed_dim")
    else:
        # try patch_embed.proj.weight shape (timm ViT style)
        try:
            proj_w = backbone.patch_embed.proj.weight
            embed_dim = proj_w.shape[0]
        except Exception:
            # try model.head input dim if available
            try:
                head_w = backbone.head.weight
                embed_dim = head_w.shape[1]  # head: [num_classes, embed_dim]
            except Exception:
                embed_dim = None

    if embed_dim is None:
        raise RuntimeError("Could not infer backbone embedding dimension. "
                           "Check the backbone API or make embed_dim available.")

    return backbone, int(embed_dim)


def get_pooled_features(backbone: nn.Module, x: torch.Tensor):
    """
    Robustly get pooled features (B, D) from a ViT-style backbone.
    Tries several common `forward_features` signatures and fallbacks.
    """
    # 1) If backbone exposes forward_features, try with common kwargs
    if hasattr(backbone, "forward_features"):
        try:
            out = backbone.forward_features(x)
            # If tuple, try to extract 2D pooled item (B, D)
            if isinstance(out, tuple):
                for item in out[::-1]:
                    if hasattr(item, "ndim") and item.ndim == 2:
                        return item
                # fallback to first element
                return out[0]
            # If tensor with ndim==3 (tokens), take cls token
            if hasattr(out, "ndim") and out.ndim == 3:
                return out[:, 0]
            return out
        except TypeError:
            # maybe accepts flags like return_all_tokens / return_pooled
            try:
                out = backbone.forward_features(x, return_all_tokens=False, return_pooled=True)
                return out
            except Exception:
                pass
        except Exception:
            pass

    # 2) Some implementations expect forward_features(x, return_all_tokens=True)
    try:
        out = backbone.forward_features(x, return_all_tokens=True)
        if hasattr(out, "ndim") and out.ndim == 3:
            return out[:, 0]
        if isinstance(out, tuple):
            for item in out[::-1]:
                if hasattr(item, "ndim") and item.ndim == 2:
                    return item
    except Exception:
        pass

    # 3) Fallback: call the full forward and try to extract pooled representation.
    # This is less ideal but keeps code robust.
    try:
        logits = backbone(x)
        # If backbone returned logits of shape (B, C) and embed_dim != C then we have no pooled features.
        # As a last resort, detach logits and return them as "features".
        return logits.detach()
    except Exception as e:
        raise RuntimeError(f"Backbone forward failed: {e}")


# -----------------------
# Trainer LightningModule
# -----------------------
class RETFoundTrainer(pl.LightningModule):
    """
    Downstream trainer that places a new classification head on top of a ViT backbone.
    The backbone can be frozen (linear probe) or unfrozen (finetune).
    """

    def __init__(self, backbone: nn.Module, lr: float, class_weights: Optional[torch.Tensor] = None, finetune: bool = False):
        super().__init__()
        # Don't attempt to save the backbone instance itself in hyperparameters.
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        # Resolve backbone and embedding dimension
        self.backbone, embed_dim = resolve_backbone(backbone)

        # Freeze backbone if requested
        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # set eval mode for backbone
            self.backbone.eval()

        # store flag
        self.finetune = finetune

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
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

    def forward(self, x: torch.Tensor):
        # Get pooled features (B, D)
        feats = get_pooled_features(self.backbone, x)
        return self.head(feats)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.acc.update(preds, y)
        self.f1.update(preds, y)

        # Explicit batch_size to avoid Lightning warning
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
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

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))

    def on_validation_epoch_end(self):
        self.log("val/acc", self.acc.compute())
        self.log("val/f1", self.f1.compute())
        self.acc.reset()
        self.f1.reset()

    def configure_optimizers(self):
        # Only parameters with requires_grad=True will be optimized
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)


# -----------------------
# Simple Eval wrapper that accepts a trained LightningModule
# -----------------------
class RETFoundEvalWrapper(pl.LightningModule):
    def __init__(self, model: pl.LightningModule):
        super().__init__()
        self.model = model.eval()

        self.acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.auroc = AUROC(task="multiclass", num_classes=NUM_CLASSES)
        self.aupr = AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
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
        except Exception:
            pass

    def on_validation_epoch_end(self):
        print("\n=== DOWNSTREAM EVALUATION ===")
        print("ACC :", self.acc.compute())
        print("F1  :", self.f1.compute())
        print("AUROC:", self.auroc.compute())
        print("AUPR:", self.aupr.compute())

        self.acc.reset()
        self.f1.reset()
        self.auroc.reset()
        self.aupr.reset()


# -----------------------
# Build RETFound (pretrained) helper
# -----------------------
def build_retfound(checkpoint_path: Optional[str] = None) -> RETFoundClassifier:
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")

    return RETFoundClassifier(
        num_classes=NUM_CLASSES,
        checkpoint_path=checkpoint_path
    ).eval()


# -----------------------
# Training function
# -----------------------
def run_retfound_training(args):
    print("\n=== RETFound Training on IDRiD ===")

    # 1. Prepare pretrained backbone (full RETFound then extract .model)
    full_model = build_retfound(args.checkpoint)
    backbone = full_model.model

    # 2. Class weights
    csv_path = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    class_weights = compute_idrid_class_weights(csv_path, NUM_CLASSES)

    # 3. Create trainer module
    model = RETFoundTrainer(backbone=backbone, lr=args.lr, class_weights=class_weights, finetune=args.finetune)

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
        patience=50,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="retfound_idrid_train"),
        callbacks=[ckpt_cb, early_cb],
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=4,
    )

    trainer.fit(model, datamodule=dm)

    print("\nSaved:", ckpt_cb.best_model_path)
    return ckpt_cb.best_model_path


# -----------------------
# Evaluation function
# -----------------------
def run_retfound_evaluation(dataset: str, checkpoint_path: Optional[str]):
    print("\n=== Evaluation on Trained Downstream Model ===")

    if checkpoint_path is None:
        raise ValueError("You must provide --checkpoint <trained.ckpt> to evaluate.")

    # 1. Load RETFound backbone (pretrained)
    full_model = build_retfound()       # loads RETFound weights
    backbone = full_model.model         # extract ViT backbone

    # 2. Load the trained downstream Lightning checkpoint and reattach backbone.
    #    load_from_checkpoint expects the same signature as __init__, so we pass backbone.
    model = RETFoundTrainer.load_from_checkpoint(
        checkpoint_path,
        backbone=backbone,
        lr=1e-6,            # placeholder (not used for eval), but kept for compatibility
        class_weights=None,
        finetune=False
    )
    model.eval()

    # 3. Data
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    if dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup(stage="validate")

    # 4. Eval wrapper + trainer
    wrapper = RETFoundEvalWrapper(model)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="retfound_downstream_eval"),
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    trainer.validate(wrapper, datamodule=dm)


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["train", "eval"])
    p.add_argument("--dataset", type=str, default="idrid", choices=["idrid", "aptos"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to RETFound pretrained weights for building backbone (train) "
                        "or path to Lightning checkpoint (eval).")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=1000)
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
