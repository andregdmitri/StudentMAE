# eval/eval_retfound.py

import torch
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC, F1Score, AveragePrecision

from models.retfound import RETFoundClassifier
from train.train_retfound import load_retfound_backbone
from dataloader.idrid import IDRiDModule
from dataloader.aptos import APTOSModule
from config.constants import *


class EvalWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

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
        preds = probs.argmax(dim=1)

        self.acc.update(preds, y)
        self.f1.update(preds, y)
        self.auroc.update(probs, y)
        self.aupr.update(probs, y)

    def on_validation_epoch_end(self):
        print("\n=== RETFound Evaluation ===")
        print("ACC :", self.acc.compute())
        print("F1  :", self.f1.compute())
        print("AUROC:", self.auroc.compute())
        print("AUPR :", self.aupr.compute())


def run_eval_retfound(args):
    print("\n=== RETFOUND EVALUATION ===")

    # Backbone
    backbone, embed_dim = load_retfound_backbone()

    # Load full model
    model = RETFoundClassifier(num_classes=NUM_CLASSES).eval()
    model.load_state_dict(torch.load(args.load_model)["state_dict"], strict=False)

    wrapper = EvalWrapper(model)

    # Dataset
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    if args.dataset == "idrid":
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup()
    else:
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
        dm.setup(stage="full")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=WandbLogger(project="retfound_eval"),
    )

    trainer.validate(wrapper, dm)
