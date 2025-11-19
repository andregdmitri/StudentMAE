# datasets/idrid.py

import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS


class IDRiDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        root = ".../versions/1/B. Disease Grading/B. Disease Grading"
        split in ["train","test"]
        """
        self.transform = transform

        # ---------------------------------------
        # image folder
        # ---------------------------------------
        img_dir = os.path.join(
            root,
            "1. Original Images",
            "a. Training Set" if split=="train" else "b. Testing Set"
        )
        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ])

        # ---------------------------------------
        # CSV path
        # ---------------------------------------
        csv_path = os.path.join(
            root,
            "2. Groundtruths",
            "a. IDRiD_Disease Grading_Training Labels.csv" if split=="train"
            else "b. IDRiD_Disease Grading_Testing Labels.csv"
        )
        df = pd.read_csv(csv_path)   # columns: Image name,Retinopathy grade,...
        df["Image name"] = df["Image name"].str.replace(".jpg","", regex=False)
        self.grade_map = dict(zip(df["Image name"], df["Retinopathy grade"]))

        print(f"Found {len(self.img_paths)} images for split={split}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img)).permute(2,0,1).float()/255.

        img_id = os.path.basename(p).replace(".jpg","")
        y = int(self.grade_map[img_id])    # 0..4

        return img, y


class IDRiDModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None):
        super().__init__()
        self.root = root
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform=transform

    def setup(self, stage=None):
        self.train = IDRiDDataset(self.root, "train", self.transform)
        self.val   = IDRiDDataset(self.root, "test",  self.transform)

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
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers
        )
