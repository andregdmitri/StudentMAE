import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BarlowTwins(pl.LightningModule):
	def __init__(self, backbone, projector_dim=128, lambda_bt=5e-3, learning_rate=1e-3):
		super().__init__()
		self.backbone = backbone
		self.projector = nn.Sequential(
			nn.Linear(backbone.output_dim, projector_dim),
			nn.BatchNorm1d(projector_dim),
			nn.ReLU(),
			nn.Linear(projector_dim, projector_dim)
		)
		self.lambda_bt = lambda_bt
		self.learning_rate = learning_rate

	def forward(self, x):
		return self.projector(self.backbone(x))

	def training_step(self, batch, batch_idx):
		(x1, x2) = batch  # Two augmented views
		z1 = self.forward(x1)
		z2 = self.forward(x2)

		# Normalize projections
		z1_norm = (z1 - z1.mean(0)) / z1.std(0)
		z2_norm = (z2 - z2.mean(0)) / z2.std(0)

		N, D = z1_norm.size()
		c = torch.mm(z1_norm.T, z2_norm) / N

		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = c.flatten()[1:].view(D - 1, D + 1)[:, :-1].flatten().pow_(2).sum()
		loss = on_diag + self.lambda_bt * off_diag
		self.log('barlow_twins_loss', loss)
		return loss

	def configure_optmizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
