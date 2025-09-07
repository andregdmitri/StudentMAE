class Logger:
    def __init__(self, log_file='training_log.txt'):
        self.log_file = log_file

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def log_training(self, epoch, loss, accuracy):
        message = f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}'
        self.log(message)

    def log_validation(self, epoch, val_loss, val_accuracy):
        message = f'Validation - Epoch: {epoch}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}'
        self.log(message)

    def log_metrics(self, metrics):
        message = 'Metrics: ' + ', '.join(f'{key}: {value:.4f}' for key, value in metrics.items())
        self.log(message)