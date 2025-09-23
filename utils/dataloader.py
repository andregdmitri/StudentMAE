import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class CustomDataset(Dataset):
	def __init__(self, data, targets):
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		x = self.data[idx]
		y = self.targets[idx]
		return x, y

class DataModule(pl.LightningDataModule):
	def __init__(self, train_data, train_targets, val_data, val_targets, batch_size=32, num_workers=0):
		super().__init__()
		self.train_data = train_data
		self.train_targets = train_targets
		self.val_data = val_data
		self.val_targets = val_targets
		self.batch_size = batch_size
		self.num_workers = num_workers
 
	def setup(self, stage=None):
		self.train_dataset = CustomDataset(self.train_data, self.train_targets)
		self.val_dataset = CustomDataset(self.val_data, self.val_targets)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
