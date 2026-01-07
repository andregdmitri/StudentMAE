# dataloader/aptos.py

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config.constants import BATCH_SIZE, NUM_WORKERS

class APTOSDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Define internal mapping for standard splits
        split_map = {
            "train": ("train_images/train_images", "train_1.csv"),
            "val":   ("val_images/val_images", "valid.csv"),
            "test":  ("test_images/test_images", "test.csv")
        }

        if split == "full":
            # Combine all known sub-directories and CSVs
            self.img_paths = []
            all_dfs = []
            for sub_dir, csv_name in split_map.values():
                d = os.path.join(root, sub_dir)
                c = os.path.join(root, csv_name)
                if os.path.exists(d):
                    self.img_paths.extend([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))])
                if os.path.exists(c):
                    all_dfs.append(pd.read_csv(c))
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            sub_dir, csv_name = split_map[split]
            img_dir = os.path.join(root, sub_dir)
            self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
            df = pd.read_csv(os.path.join(root, csv_name))

        # Cleanup ID codes and create label map
        df["id_code"] = df["id_code"].str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False)
        self.label_map = dict(zip(df["id_code"], df["diagnosis"]))
        print(f"[APTOS] Split: {split:<5} | Images: {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        img_id = os.path.basename(path).split(".")[0]
        label = int(self.label_map.get(img_id, 0)) 
        return img, label, path

class APTOSModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        # stage can be 'fit', 'validate', 'test', or 'predict'
        if stage == "fit" or stage is None:
            self.train_ds = APTOSDataset(self.root, split="train", transform=self.transform)
            self.val_ds   = APTOSDataset(self.root, split="val",   transform=self.transform)
        if stage == "test":
            self.test_ds  = APTOSDataset(self.root, split="test",  transform=self.transform)
        if stage == "full":
            self.full_ds  = APTOSDataset(self.root, split="full",  transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def full_dataloader(self):
        # Useful for extracting features from the entire dataset
        ds = APTOSDataset(self.root, split="full", transform=self.transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)