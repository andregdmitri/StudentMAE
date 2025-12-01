# train/head.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule, compute_aptos_class_weights
from torchvision import transforms
import os


class HeadTrainer(pl.LightningModule):
    """
    Phase II trainer:
    - Loads frozen backbone
    - Trains classification head only
    """

    def __init__(self, backbone, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.eval()

        # trainable head
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

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
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
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
        return self.head(feats)

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
        self.log("val/auroc", auroc)
        self.log("val/aupr", aupr)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_aupr.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )
    
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
        use_cls_token=False,
    )

    ckpt = torch.load(args.load_backbone, map_location="cpu")

    if "backbone" in ckpt:
        backbone.load_state_dict(ckpt["backbone"])
    else:
        backbone.load_state_dict(ckpt)

    # ------------------------------
    # 2. Build HeadTrainer
    # ------------------------------
    csv_path = os.path.join(
        IDRID_PATH,
        "2. Groundtruths",
        "a. IDRiD_Disease Grading_Training Labels.csv"
    )
    class_weights = compute_idrid_class_weights(csv_path, NUM_CLASSES)
    print("Class weights:", class_weights)
    lr = args.head_lr if args.head_lr else HEAD_LR
    model = HeadTrainer(backbone, lr, class_weights=class_weights)

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
    # 4. Callbacks
    # ------------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="best_head",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        patience=50,
        mode="max",
        verbose=False
    )

    # ------------------------------
    # 5. Trainer
    # ------------------------------
    epochs = HEAD_EPOCHS

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=4,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # ------------------------------
    # 6. Train
    # ------------------------------
    trainer.fit(model, datamodule=dm)

    # ------------------------------
    # 7. Save final backbone + head
    # ------------------------------
    save_path = os.path.join(CHECKPOINT_DIR, "vmamba_final_head.pth")
    ckpt = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
    state = ckpt["state_dict"]  # ← extract real weights

    # Split backbone & head cleanly
    backbone_state = {}
    head_state = {}

    for k, v in state.items():
        if k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
        elif k.startswith("head."):
            head_state[k[len("head."):]] = v

    final = {
        "backbone": backbone_state,
        "head": head_state,
    }

    torch.save(final, save_path)
    print(f"[✓] Saved final model → {save_path}")