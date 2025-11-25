# constants.py
import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Model Config
DIST_LR = float(os.getenv('DIST_LR', 1e-4))
HEAD_LR = float(os.getenv('HEAD_LR', 5e-3))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
DIST_EPOCHS = int(os.getenv('DIST_EPOCHS', 1000))
HEAD_EPOCHS = int(os.getenv('HEAD_EPOCHS', 1000))
MASK_RATIO = float(os.getenv('MASK_RATIO', 0.0))

INPUT_DIM = int(os.getenv('INPUT_DIM', 784))
NUM_CLASSES = int(os.getenv('NUM_CLASSES', 5))
IMG_SIZE = int(os.getenv('IMG_SIZE', 224))
PATCH_SIZE = int(os.getenv('PATCH_SIZE', 16))
IN_CHANS = int(os.getenv('IN_CHANS', 3))
VMAMBA_DEPTH = int(os.getenv('VMAMBA_DEPTH', 8))
VMAMBA_EMBED_DIM = int(os.getenv('VMAMBA_EMBED_DIM', 64))
SSM_DIM = int(os.getenv('SSM_DIM', 8))
EXPAND_DIM = int(os.getenv('EXPAND_DIM', VMAMBA_EMBED_DIM * 2))
TEACHER_EMBED_DIM = int(os.getenv('TEACHER_EMBED_DIM', 1024))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', 12))
PATIENCE = int(os.getenv('PATIENCE', 15))

# Other settings
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = os.getenv('DATA_DIR', './data')
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
IDRID_PATH = os.getenv('IDRID_PATH', os.path.join(DATA_DIR, 'aaryapatel98/indian-diabetic-retinopathy-image-dataset/versions/1/B.%20Disease%20Grading/B. Disease Grading'))