from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

class BaseClassificationModel(LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        super(BaseClassificationModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Define the model architecture
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dim)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc)

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