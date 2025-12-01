# dataloader/aptos.py

import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS


class APTOSDataset(Dataset):
    """
    APTOS2019 dataset.
    split in ["train", "val", "test", "full"]
    """

    def __init__(self, root, split="full", transform=None):
        self.transform = transform
        self.split = split

        # --------------------------------------------------
        # FULL SPLIT (short-circuit early)
        # --------------------------------------------------
        if split == "full":
            img_dirs = [
                os.path.join(root, "train_images", "train_images"),
                os.path.join(root, "val_images", "val_images"),
                os.path.join(root, "test_images", "test_images"),
            ]

            self.img_paths = []
            for d in img_dirs:
                if not os.path.exists(d):
                    continue
                self.img_paths.extend([
                    os.path.join(d, f)
                    for f in os.listdir(d)
                    if f.lower().endswith((".png", ".jpg"))
                ])

            # merge CSVs
            df_train = pd.read_csv(os.path.join(root, "train_1.csv"))
            df_val   = pd.read_csv(os.path.join(root, "valid.csv"))
            df_test  = pd.read_csv(os.path.join(root, "test.csv"))

            df = pd.concat([df_train, df_val, df_test], ignore_index=True)
            df["id_code"] = df["id_code"].str.replace(".png","",regex=False)\
                                         .str.replace(".jpg","",regex=False)

            self.label_map = dict(zip(df["id_code"], df["diagnosis"]))

            print(f"[APTOS] Found {len(self.img_paths)} FULL images")
            return  # 🔥 IMPORTANT: avoid falling through

        # --------------------------------------------------
        # STANDARD SPLITS
        # --------------------------------------------------
        if split == "train":
            img_dir = os.path.join(root, "train_images", "train_images")
            csv_path = os.path.join(root, "train_1.csv")

        elif split == "val":
            img_dir = os.path.join(root, "val_images", "val_images")
            csv_path = os.path.join(root, "valid.csv")

        elif split == "test":
            img_dir = os.path.join(root, "test_images", "test_images")
            csv_path = os.path.join(root, "test.csv")

        else:
            raise ValueError("Invalid split: must be train/val/test/full")

        # list images
        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])

        # load CSV
        df = pd.read_csv(csv_path)
        df["id_code"] = df["id_code"].str.replace(".png","",regex=False)\
                                     .str.replace(".jpg","",regex=False)
        self.label_map = dict(zip(df["id_code"], df["diagnosis"]))

        print(f"[APTOS] Found {len(self.img_paths)} images for split={split}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.

        img_id = os.path.basename(p).split(".")[0]
        y = int(self.label_map[img_id])  # diagnosis is 0..4

        return img, y, p


class APTOSModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = APTOSDataset(self.root, split="train", transform=self.transform)
            self.val   = APTOSDataset(self.root, split="val",   transform=self.transform)

        # "full" mode: full validation set for inference/eval
        if stage == "validate" or stage == "test" or stage == "predict":
            self.full = APTOSDataset(self.root, split="full", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        if hasattr(self, "full"):
            return DataLoader(self.full, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def predict_dataloader(self):
        return self.val_dataloader()



def compute_aptos_class_weights(csv_path, num_classes=5):
    df = pd.read_csv(csv_path)
    counts = df["diagnosis"].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=0)

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)
