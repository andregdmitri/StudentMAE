import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision
from torchvision import transforms

from config.constants import *
from models.retfound import RETFoundBackbone
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule
from optimizers.optmizer import warmup_cosine_optimizer

# -----------------------------------------------------------
#  Module: RETFoundTask
# -----------------------------------------------------------

class RETFoundTask(pl.LightningModule):
    def __init__(self, mode="linear", lr=3e-4, checkpoint_path=None, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.mode = mode
        
        # 1. Load the RETFound Model
        # We use the existing RETFoundBackbone to get the ViT backbone
        full_model = RETFoundBackbone(
            num_classes=NUM_CLASSES,
            checkpoint_path=checkpoint_path or os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")
        )
        self.backbone = full_model.model
        self.embed_dim = self.backbone.head.weight.shape[1]
        
        # 2. Re-initialize Head
        self.head = nn.Linear(self.embed_dim, NUM_CLASSES)
        
        # 3. Handle Freezing logic
        if mode == "linear":
            for param in self.backbone.parameters():
                param.requires_grad = False
        else: # finetune
            for param in self.backbone.parameters():
                param.requires_grad = True

        # 4. Loss & Metrics
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        
        self.metrics = nn.ModuleDict({
            "acc": Accuracy(task="multiclass", num_classes=NUM_CLASSES),
            "f1": F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            "auroc": AUROC(task="multiclass", num_classes=NUM_CLASSES),
            "aupr": AveragePrecision(task="multiclass", num_classes=NUM_CLASSES)
        })

    def forward(self, x):
        # RETFound ViT-L forward_features returns pooled [CLS] or average features (B, D)
        if self.mode == "linear":
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

        # Update metrics
        self.metrics["acc"].update(preds, y)
        self.metrics["f1"].update(preds, y)
        try:
            self.metrics["auroc"].update(probs, y)
            self.metrics["aupr"].update(probs, y)
        except ValueError: 
            pass # Handle cases where a batch doesn't contain all classes

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_train_epoch_end(self):
        self._log_and_reset_metrics("train")

    def on_validation_epoch_end(self):
        self._log_and_reset_metrics("val")

    def _log_and_reset_metrics(self, stage):
        for name, metric in self.metrics.items():
            try:
                val = metric.compute()
                self.log(f"{stage}/{name}", val, prog_bar=(name == "f1" or name == "acc"))
            except:
                pass
            metric.reset()

    def configure_optimizers(self):
        if self.mode == "linear":
            # Simple AdamW for Linear Probing
            return torch.optim.AdamW(self.head.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        else:
            # Warmup Cosine for Fine-tuning
            optimizer, scheduler = warmup_cosine_optimizer(
                parameters=self.parameters(),
                max_epochs=self.trainer.max_epochs,
                lr=self.hparams.lr,
                warmup_epochs=WARMUP_EPOCHS,
                final_lr=FINAL_LR,
                weight_decay=WEIGHT_DECAY
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

# -----------------------------------------------------------
#  Unified Train Entry
# -----------------------------------------------------------

def run_train_retfound(args):
    pl.seed_everything(args.seed or 42)
    
    # 1. Setup Data
    # Use standard resize for finetune, stronger augs for linear probing
    from utils.transforms import train_transform_retfound_linear, train_transform_default

    if args.retfound_mode == "linear":
        tfm = train_transform_retfound_linear(IMG_SIZE)
    else:
        tfm = train_transform_default(IMG_SIZE)

    if args.dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
        csv = os.path.join(IDRID_PATH, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
        class_weights = compute_idrid_class_weights(csv, NUM_CLASSES)
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        class_weights = None

    dm.setup()

    # 2. Setup Model
    model = RETFoundTask(
        mode=args.retfound_mode, 
        lr=args.lr, 
        checkpoint_path=args.checkpoint,
        class_weights=class_weights
    )

    # 3. Trainer
    ckpt_cb = ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename=f"retfound_{args.retfound_mode}_best")
    early_cb = EarlyStopping(monitor="val/f1", patience=50, mode="max")
    
    import time
    run_name = f"retfound_{args.retfound_mode}_{args.dataset}_{int(time.time())}"
    logger = WandbLogger(project="retfound_unified", name=run_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[ckpt_cb, early_cb],
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, dm)
    print(f"\n[✓] Training Complete. Best model: {ckpt_cb.best_model_path}")

    # load best checkpoint into model and compute metrics on train & val
    try:
        best_ckpt = torch.load(ckpt_cb.best_model_path, map_location="cpu").get("state_dict", {})
        model.load_state_dict(best_ckpt, strict=False)
        dm.setup()
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        val_metrics = trainer.validate(model, dataloaders=val_dl)
        train_metrics = trainer.validate(model, dataloaders=train_dl)
    except Exception:
        val_metrics = []
        train_metrics = []

    try:
        from utils.results import append_result
        row = {
            "timestamp": int(time.time()),
            "mode": f"retfound_{args.retfound_mode}",
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