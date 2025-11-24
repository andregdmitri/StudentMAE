# train.distill.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from config.constants import *
from models.retfound import RETFoundClassifier
from models.vmamba_backbone import VisualMamba
from models.dist import DistillationModule
from dataloader.idrid import IDRiDModule
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import torch.nn as nn

def build_student(args):
    lr = args.dist_lr if args.dist_lr is not None else DIST_LR
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else MASK_RATIO

    return VisualMamba(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        learning_rate=lr,
        mask_ratio=mask_ratio
    )


def build_teacher():
    return RETFoundClassifier(
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")
    ).eval()  # teacher fully frozen


def run_distillation(args):
    pl.seed_everything(42)
    logger = WandbLogger(
        project="vmamba_distillation",
        log_model=True,
        save_dir="wandb_logs"
    )

    print("=== PHASE I: DISTILLATION ===")

    # Teacher & student
    teacher = build_teacher()
    student = build_student(args)

    # Project student -> teacher dimension
    projector = nn.Sequential(
        nn.Linear(VMAMBA_EMBED_DIM, 2 * VMAMBA_EMBED_DIM),
        nn.GELU(),
        nn.Linear(2 * VMAMBA_EMBED_DIM, TEACHER_EMBED_DIM),
    )

    # Data
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dm = IDRiDModule(
        root=IDRID_PATH,
        transform=tfm,
        batch_size=BATCH_SIZE
    )
    dm.setup()

    # Distillation model
    lr = args.dist_lr if args.dist_lr else DIST_LR
    epochs = args.dist_epochs if args.dist_epochs is not None else DIST_EPOCHS

    model = DistillationModule(
        teacher=teacher,
        student=student,
        projector=projector,
        lr=lr,
    )


    # ------------------------------
    # Early stopping callback
    # ------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val/distill_loss",
        patience=50,
        mode="min",
        min_delta=0.0,   
        verbose=False
    )

    # ------------------------------
    # Checkpoint callback
    # ------------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val/distill_loss",
        mode="min",
        save_top_k=1,
        filename="best_distillation",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=4,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, datamodule=dm)

    # ------------------------------
    # Save distilled student + projector separately
    # ------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    student_path = os.path.join(CHECKPOINT_DIR, "vmamba_distilled_student.pth")
    projector_path = os.path.join(CHECKPOINT_DIR, "vmamba_distilled_projector.pth")

    torch.save({"backbone": student.state_dict()}, student_path)
    torch.save({"projector": projector.state_dict()}, projector_path)

    print("\n========================")
    print("Saved distilled backbone:")
    print("  Student:   ", student_path)
    print("  Projector: ", projector_path)
    print("========================\n")

