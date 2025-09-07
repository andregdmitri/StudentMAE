import torch
from torch.utils.data import DataLoader
from neural_network.base_classification_model import BaseClassificationModel
from utils.logger import Logger
from utils.metrics import accuracy
from config.constants import BATCH_SIZE, LEARNING_RATE

def evaluate_student(model_path, data_loader):
    model = BaseClassificationModel.load_from_checkpoint(model_path)
    model.eval()
    
    logger = Logger()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy_score = total_correct / total_samples
    logger.log(f'Accuracy: {accuracy_score:.4f}')
    return accuracy_score

if __name__ == "__main__":
    # Example usage
    from data import MyDataset  # Assuming MyDataset is defined in data folder
    dataset = MyDataset()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model_path = 'path/to/your/model.ckpt'  # Update with your model path
    evaluate_student(model_path, data_loader)