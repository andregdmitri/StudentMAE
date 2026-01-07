import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms

from config.constants import *
from models.retfound import RETFoundBackbone
from models.vmamba_backbone import VisualMamba
from models.dist import DistillationModule
from dataloader.idrid import IDRiDModule
from dataloader.aptos import APTOSModule

# -----------------------------------------------------------
#  Helpers: Model Builders
# -----------------------------------------------------------

def build_student(args):
    """Initializes the VMamba student backbone."""
    return VisualMamba(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        learning_rate=args.lr or LR,
        mask_ratio=args.mask_ratio or MASK_RATIO,
        use_cls_token=False
    )

def build_teacher():
    """Initializes and freezes the RETFound teacher."""
    path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")
    teacher = RETFoundBackbone(checkpoint_path=path)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

# -----------------------------------------------------------
#  Unified Distillation Entry
# -----------------------------------------------------------

def run_distillation(args):
    pl.seed_everything(42)
    print("\n=== PHASE I: VMAMBA DISTILLATION ===")

    # 1. Models
    teacher = build_teacher()
    student = build_student(args)

    # Project student output dim (VMAMBA_EMBED_DIM) to teacher dim (1024)
    # This is often needed if the student is smaller than the teacher.
    projector = nn.Sequential(
        nn.Linear(VMAMBA_EMBED_DIM, 2 * VMAMBA_EMBED_DIM),
        nn.GELU(),
        nn.Linear(2 * VMAMBA_EMBED_DIM, TEACHER_EMBED_DIM),
    )

    # 2. Data
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == "aptos":
        dm = APTOSModule(root=APTOS_PATH, transform=tfm, batch_size=BATCH_SIZE)
    else:
        dm = IDRiDModule(root=IDRID_PATH, transform=tfm, batch_size=BATCH_SIZE)
    dm.setup()

    # 3. Distillation Wrapper
    # This LightningModule handles the loss calculation (MSE/Cosine) between teacher and student
    model = DistillationModule(
        teacher=teacher,
        student=student,
        projector=projector,
        lr=args.lr or LR,
    )

    # 4. Callbacks & Trainer
    ckpt_cb = ModelCheckpoint(
        monitor="val/distill_loss",
        mode="min",
        save_top_k=1,
        filename="best_distillation"
    )
    
    early_cb = EarlyStopping(
        monitor="val/distill_loss",
        patience=PATIENCE,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.dist_epochs or DIST_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        logger=WandbLogger(project="vmamba_distillation"),
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=5
    )

    # 5. Run Distillation
    trainer.fit(model, datamodule=dm)

    # 6. Final Extraction & Saving
    save_backbone_and_projector(ckpt_cb.best_model_path)


def save_backbone_and_projector(best_ckpt_path):
    """Extracts student and projector weights from the unified Lightning checkpoint."""
    print(f"\n[i] Extracting weights from: {best_ckpt_path}")
    
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # Extracting by prefixing logic
    student_sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")}
    projector_sd = {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    student_path = os.path.join(CHECKPOINT_DIR, "vmamba_distilled_student.pth")
    projector_path = os.path.join(CHECKPOINT_DIR, "vmamba_distilled_projector.pth")

    # Saving in the format required by the 'Head' training Phase II
    torch.save({"backbone": student_sd}, student_path)
    torch.save({"projector": projector_sd}, projector_path)

    print(f"[✓] Saved Student Backbone: {student_path}")
    print(f"[✓] Saved Projector:        {projector_path}\n")