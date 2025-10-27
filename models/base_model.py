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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def evaluate(self, dataloader):
        self.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                total_loss += loss.item()
                
                preds = torch.argmax(y_hat, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy