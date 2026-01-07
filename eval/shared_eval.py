import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from config.constants import NUM_CLASSES

class EvalWrapper(pl.LightningModule):
    """
    Standardized evaluation wrapper for any model (VMamba or RETFound).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

        # Initialize standard metrics
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
        
        # AUROC/AUPR can fail if a batch is too small or missing classes
        try:
            self.auroc.update(probs, y)
            self.aupr.update(probs, y)
        except ValueError:
            pass

    def on_validation_epoch_end(self):
        # Compute metrics
        results = {
            "eval/acc": self.acc.compute(),
            "eval/f1": self.f1.compute(),
            "eval/auroc": self.auroc.compute(),
            "eval/aupr": self.aupr.compute()
        }

        # Log to Logger (Wandb)
        self.log_dict(results)

        # Print to Console
        print("\n" + "="*30)
        print(" EVALUATION RESULTS")
        print("="*30)
        for k, v in results.items():
            print(f"{k.split('/')[-1].upper():<10}: {v:.4f}")
        print("="*30 + "\n")

        # Reset for next run
        self.acc.reset()
        self.f1.reset()
        self.auroc.reset()
        self.aupr.reset()