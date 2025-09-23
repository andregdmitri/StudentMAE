# constants.py
import os
import torch
from dotenv import load_dotenv

load_dotenv()

LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPOCHS = int(os.getenv('EPOCHS', 50))
INPUT_DIM = int(os.getenv('INPUT_DIM', 784))
NUM_CLASSES = int(os.getenv('NUM_CLASSES', 10))
DROPOUT_RATE = float(os.getenv('DROPOUT_RATE', 0.5))
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.getenv('DATA_DIR', os.path.join(os.getcwd(), 'data'))