import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


@torch.no_grad()
def get_teacher_embedding(model, x):
    model.eval()
    feats = model.forward_features(x)
    return feats.mean(dim=1) if feats.dim() == 3 else feats


def get_student_embedding(model, x):
    model.train()   # student is being trained
    return model.forward_features(x)    # this is correct path


def distill_embeddings(teacher, student, projector, dataloader, optimizer, device, epochs=10):
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)

            with torch.no_grad():
                teacher_emb = get_teacher_embedding(teacher, batch)      # [B,1024]

            student_emb = projector(get_student_embedding(student, batch))   # [B,1024]

            loss = criterion(student_emb, teacher_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"epoch {epoch+1}  loss = {epoch_loss/len(dataloader):.6f}")
