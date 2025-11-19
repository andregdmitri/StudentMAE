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
from pytorch_lightning.callbacks import ModelCheckpoint
import os

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
    ).eval()


def run_distillation(args):
    pl.seed_everything(42)
    logger = WandbLogger(project="vmamba_distillation", log_model=True, save_dir="wandb_logs")

    print("=== PHASE I: DISTILLATION ===")
    teacher = build_teacher()
    student = build_student(args)
    projector = torch.nn.Linear(VMAMBA_EMBED_DIM, TEACHER_EMBED_DIM)

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

    lr = args.dist_lr if args.dist_lr else DIST_LR
    epochs = args.dist_epochs if args.dist_epochs is not None else DIST_EPOCHS
    
    model = DistillationModule(
        teacher=teacher,
        student=student,
        projector=projector,
        lr=lr
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best_distillation",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=dm)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # If checkpoint callback saved a model, copy it; also save student's state_dict separately
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        print("Best checkpoint:", best_ckpt)
        # Optionally load best ckpt to restore model weights into student
        try:
            ckpt = torch.load(best_ckpt, map_location="cpu")
            # If saved checkpoint contains student state_dict directly, adjust accordingly.
        except Exception:
            pass

    # Save student backbone weights (final)
    save_path = os.path.join(CHECKPOINT_DIR, "vmamba_distilled.pth")
    torch.save(student.state_dict(), save_path)
    print("Saved student to:", save_path)