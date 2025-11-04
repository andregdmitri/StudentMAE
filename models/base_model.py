from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

from config.constants import *

class BaseClassifier(LightningModule):
    def __init__(self, 
                 input_dim=INPUT_DIM, 
                 num_classes=NUM_CLASSES, 
                 learning_rate=LEARNING_RATE
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
        # Must be implemented by subclass
        self.model = None
    
    def forward(self, x):
        if self.model is None:
            raise NotImplementedError("Child class must define its own forward pass")
        return self.model(x)

    def _compute_loss(self, out, y_dict):
        total = 0.0
        for task, logits in out.items():
            # logits: (B,1)
            y = y_dict[task].float().view(-1,1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total += loss
        return total

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        if isinstance(out, dict):        # multi-task
            loss = self._compute_loss(out, y)
            self.log("train_total_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        else:                             # single task
            loss = F.cross_entropy(out, y)
            self.log("train_loss", loss, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        if isinstance(out, dict):
            _ = self._compute_loss(out, y)
            self.log("val_total_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = F.cross_entropy(out, y)
            self.log("val_total_loss", loss, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def evaluate(self, dataloader):
        self.eval()
        total_loss = 0; total_samples = 0
        with torch.no_grad():
            for x,y in dataloader:
                out = self(x)
                batch_loss = 0
                for task,logits in out.items():
                    batch_loss += F.binary_cross_entropy_with_logits(
                        logits, y[task].float().view(-1,1)
                    ).item()
                total_loss += batch_loss
                total_samples += len(x)
        avg_loss = total_loss / total_samples
        return avg_loss