# dist.py

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
    return feats


def get_student_embedding(model, x):
    model.train()
    return model.forward_features(x)


def get_student_backbone_params(student):
    for name, p in student.named_parameters():
        if not name.startswith("heads."):
            yield p


# -------------------------------
# Phase 1: foundation distillation
# -------------------------------
def distill_embeddings(teacher, student, projector, dataloader, device, epochs=10, lr=1e-4):
    criterion = nn.CosineEmbeddingLoss()

    optimizer = torch.optim.Adam(
        list(get_student_backbone_params(student)) + list(projector.parameters()),
        lr=lr
    )

    teacher.eval()

    for epoch in range(epochs):
        epoch_loss = 0.

        for x, _ in dataloader:
            x = x.to(device)

            with torch.no_grad():
                t = get_teacher_embedding(teacher, x)    # [B,1024]

            s = projector(get_student_embedding(student, x))  # [B,1024]

            target = torch.ones(s.size(0)).to(device)
            loss = criterion(s, t, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Distill] epoch {epoch+1}  loss={epoch_loss/len(dataloader):.6f}")


# -------------------------------------
# Phase 2: train head w/ real labels
# -------------------------------------
def train_head(student, dataloader, device, epochs=10, lr=1e-4):

    # freeze backbone params
    for p in get_student_backbone_params(student):
        p.requires_grad_(False)

    # unfreeze classifier head
    for p in student.head.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(student.head.parameters(), lr=lr)

    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).long()         # [B]

            logits = student(x)             # [B,5]
            loss = ce(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Head] epoch {epoch+1}  loss={epoch_loss/len(dataloader):.6f}")