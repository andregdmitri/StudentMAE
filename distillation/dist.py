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
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        list(get_student_backbone_params(student)) +
        list(projector.parameters()),
        lr=lr
    )

    # freeze heads:
    for p in student.heads.parameters():
        p.requires_grad_(False)

    for epoch in range(epochs):
        epoch_loss = 0.

        for batch in dataloader:
            batch = batch.to(device)

            with torch.no_grad():
                t = get_teacher_embedding(teacher, batch)  # [B,D_teacher]

            s = projector(get_student_embedding(student, batch))         # [B,D_teacher]

            loss = criterion(s, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Distill] epoch {epoch+1}  loss={epoch_loss/len(dataloader):.6f}")


# -------------------------------------
# Phase 2: train heads w/ real labels
# -------------------------------------
def train_heads(student, dataloader, device, epochs=10, lr=1e-4):
    """
    student : VisualMamba (already distilled)
    dataloader : should return (x, y_dict)
    """

    # freeze backbone:
    for p in get_student_backbone_params(student):
        p.requires_grad_(False)

    # unfreeze heads:
    for p in student.heads.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        student.heads.parameters(),
        lr=lr
    )

    student.train()

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        epoch_loss = 0.

        for x, y_dict in dataloader:
            x = x.to(device)
            for k in y_dict:
                y_dict[k] = y_dict[k].to(device).float().view(-1,1)

            out = student(x)       # dict[task: logits]

            loss = 0.
            for task, logits in out.items():
                loss += bce(logits, y_dict[task])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Heads] epoch {epoch+1}  loss={epoch_loss/len(dataloader):.6f}")
