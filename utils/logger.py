import wandb

class Logger:
    def __init__(self, project='StudentMAE', run_name=None):
        self.run = wandb.init(project=project, name=run_name)

    def log_training(self, epoch, loss, accuracy):
        wandb.log({'train/epoch': epoch, 'train/loss': loss, 'train/accuracy': accuracy})

    def log_validation(self, epoch, val_loss, val_accuracy):
        wandb.log({'val/epoch': epoch, 'val/loss': val_loss, 'val/accuracy': val_accuracy})

    def log_metrics(self, metrics):
        wandb.log(metrics)