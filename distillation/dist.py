
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.retfound import RETFoundClassifier 
from models.vmamba import VisualMamba

# Dummy dataset class (replace with your actual dataset)
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def get_teacher_embedding(model, x):
    """
    Extracts the embedding from the RETFound VisionTransformer backbone (pre-classifier).
    Handles both pooled and token-level outputs.
    """
    with torch.no_grad():
        vit = model.model  # VisionTransformer inside RETFoundClassifier

        # Ensure we don't pool inside the transformer
        if hasattr(vit, "global_pool"):
            vit.global_pool = False

        # Forward through patch embedding and transformer blocks
        feats = vit.forward_features(x)  # should return [B, N, D] or [B, D]

        # Handle different return shapes
        if feats.dim() == 4:
            # Something went wrong — still an image
            raise RuntimeError(
                f"Teacher ViT returned {feats.shape}, expected [B, N, D] or [B, D]. "
                "Make sure you're calling vit.forward_features(), not the full classifier."
            )
        elif feats.dim() == 3:
            emb = feats.mean(dim=1)  # Mean over tokens
        elif feats.dim() == 2:
            emb = feats  # Already pooled
        else:
            raise RuntimeError(f"Unexpected teacher embedding shape: {feats.shape}")

    return emb

def get_student_embedding(model, x):
    """
    Extracts the embedding from the student (VMamba) before the classification head.
    """
    # Patch embedding
    x = model.patch_embed(x)
    x = x.flatten(2).transpose(1, 2)
    if model.mask_ratio > 0.0:
        x, _ = model.apply_mask(x)
    x = model.backbone(x)
    emb = x.mean(dim=1)
    return emb

def distill_embeddings(teacher, student, projector, dataloader, optimizer, device, epochs=10):
    teacher.eval()
    student.train()
    projector.train()
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs = batch.to(device)

            with torch.no_grad():
                teacher_emb = get_teacher_embedding(teacher, inputs)

            student_emb = get_student_embedding(student, inputs)
            student_emb = projector(student_emb)

            loss = criterion(student_emb, teacher_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # print(f"Student emb: {student_emb.shape}, Teacher emb: {teacher_emb.shape}")

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.6f}")