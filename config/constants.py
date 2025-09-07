# constants.py

LEARNING_RATE =  os.getenv("DATA_DIR", "./data/images")
BATCH_SIZE = 32
EPOCHS = 50
INPUT_DIM = 784  # Example for MNIST
NUM_CLASSES = 10  # Example for MNIST
DROPOUT_RATE = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'