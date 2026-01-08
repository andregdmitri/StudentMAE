import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
from torchvision import transforms

from utils.transforms import eval_transform
from config.constants import *
from models.vmamba_backbone import VisualMamba
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule
from optimizers.optimizer import warmup_cosine_optimizer

class VMambaHeadTask(pl.LightningModule):
    def __init__(self, backbone, lr, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])
        self.backbone = backbone

        # 1. Handle Freezing logic
        if FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        # 2. Trainable head (Architecture kept identical to your original)
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

        # 3. Loss & Metrics
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        
        # Modularized metrics
        metric_args = {"task": "multiclass", "num_classes": NUM_CLASSES}
        self.metrics = nn.ModuleDict({
            "acc": Accuracy(**metric_args),
            "f1": F1Score(**metric_args, average="macro"),
            "auroc": AUROC(**metric_args),
            "aupr": AveragePrecision(**metric_args)
        })

    def forward(self, x):
        # Apply no_grad only if frozen to save memory during training
        if FREEZE_BACKBONE and not self.training:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)
        return self.head(feats)

    def shared_step(self, batch, stage):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Update metrics (safe-guarded for AUROC)
        self.metrics["acc"].update(preds, y)
        self.metrics["f1"].update(preds, y)
        try:
            self.metrics["auroc"].update(probs, y)
            self.metrics["aupr"].update(probs, y)
        except ValueError: pass 

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_train_epoch_end(self): self._log_metrics("train")
    def on_validation_epoch_end(self): self._log_metrics("val")

    def _log_metrics(self, stage):
        for name, m in self.metrics.items():
            try:
                self.log(f"{stage}/{name}", m.compute(), prog_bar=(name == "f1"))
            except: pass
            m.reset()

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
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# -----------------------------------------------------------
#  Training Entry Point
# -----------------------------------------------------------

def run_head_training(args):
    pl.seed_everything(args.seed or 42)
    print("\n=== PHASE II: VMAMBA HEAD TRAINING ===")

    # 1. Backbone Loading logic
    backbone = VisualMamba(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM, depth=VMAMBA_DEPTH, 
        learning_rate=0.0, mask_ratio=0.0, use_cls_token=False,
    )

    ckpt = torch.load(args.load_backbone, map_location="cpu")
    # Handle both Lightning state_dicts and raw state_dicts
    state_dict = ckpt.get("state_dict", ckpt)
    new_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone." in k}
    backbone.load_state_dict(new_state if new_state else state_dict, strict=False)

    # 2. Data & Weights
    tfm = eval_transform(IMG_SIZE)
    if args.dataset == "aptos":
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        class_weights = None
    else:
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
        csv_path = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
        class_weights = compute_idrid_class_weights(csv_path, NUM_CLASSES)

    dm.setup()

    # 3. Model & Trainer
    model = VMambaHeadTask(backbone, lr=args.lr or LR, class_weights=class_weights)
    
    ckpt_cb = ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename="best_head")
    early_cb = EarlyStopping(monitor="val/f1", patience=100, mode="max")

    import time
    run_name = f"head_{args.dataset}_{int(time.time())}"
    trainer = pl.Trainer(
        max_epochs=HEAD_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="vmamba_head_training", name=run_name),
        precision="16-mixed",
        callbacks=[ckpt_cb, early_cb]
    )

    trainer.fit(model, dm)

    # 4. Final Save (Backbone + Head split)
    save_path = os.path.join(CHECKPOINT_DIR, "vmamba_final_head.pth")
    best_ckpt = torch.load(ckpt_cb.best_model_path)["state_dict"]

    final_dict = {
        "backbone": {k.replace("backbone.", ""): v for k, v in best_ckpt.items() if k.startswith("backbone.")},
        "head": {k.replace("head.", ""): v for k, v in best_ckpt.items() if k.startswith("head.")}
    }
    torch.save(final_dict, save_path)
    print(f"[✓] Model saved to {save_path}")

    # load best weights into model and compute metrics
    try:
        model.load_state_dict(best_ckpt, strict=False)
        dm.setup()
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        val_metrics = trainer.validate(model, dataloaders=val_dl)
        train_metrics = trainer.validate(model, dataloaders=train_dl)
    except Exception:
        val_metrics = []
        train_metrics = []

    # append to results csv
    try:
        from utils.results import append_result
        import time
        row = {
            "timestamp": int(time.time()),
            "mode": "head",
            "dataset": args.dataset,
            "model_path": ckpt_cb.best_model_path,
            "run_name": run_name,
            "seed": args.seed,
            "monitor": ckpt_cb.monitor if hasattr(ckpt_cb, "monitor") else None,
            "monitor_value": float(ckpt_cb.best_model_score) if getattr(ckpt_cb, "best_model_score", None) is not None else None,
            "train_metrics": train_metrics[0] if train_metrics else {},
            "val_metrics": val_metrics[0] if val_metrics else {},
        }
        append_result(row)
    except Exception:
        pass