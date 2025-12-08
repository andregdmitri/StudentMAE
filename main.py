# main.py

import argparse

from config.constants import (
    LR,
    MASK_RATIO,
    DIST_EPOCHS,
    HEAD_EPOCHS,
)
from train.distill import run_distillation
from train.head import run_head_training
from eval.eval_vmamba import run_evaluation

from train.train_retfound import run_train_retfound
from eval.eval_retfound import run_eval_retfound


def parse_args():
    epilog = """
Examples:

  # -----------------------------
  # Phase I: Distillation
  # -----------------------------
  python main.py --run distill \
      --lr 1e-4 \
      --mask_ratio 0.75

  # -----------------------------
  # Phase II: Head training
  # -----------------------------
  python main.py --run head \
      --load_backbone checkpoints/vmamba_distilled_student.pth \
      --lr 1e-4

  # -----------------------------
  # Phase III: Evaluation
  # -----------------------------
  python main.py --run eval \
      --load_model checkpoints/vmamba_final_head.pth

  # -----------------------------
  # RETFound Linear Probe
  # -----------------------------
  python main.py --run retfound_linear \
      --dataset idrid \
      --lr 3e-4

  # -----------------------------
  # RETFound Fine-Tuning
  # -----------------------------
  python main.py --run retfound_finetune \
      --dataset aptos \
      --lr 1e-5

  # -----------------------------
  # RETFound Evaluation
  # -----------------------------
  python main.py --run retfound_eval \
      --load_model retfound_finetuned.pth
"""

    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline for VMamba + RETFound",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ----------------------------------------------------------------------
    # MODE SELECTION
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--run", type=str, required=True,
        choices=[
            "distill",
            "head",
            "eval",
            "retfound_linear",
            "retfound_finetune",
            "retfound_eval",
        ],
        help="Which pipeline to run"
    )

    # ----------------------------------------------------------------------
    # SHARED ARGUMENTS
    # ----------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate for distillation & head training (and RETFound)")

    parser.add_argument("--mask_ratio", type=float, default=MASK_RATIO,
                        help="Mask ratio for student MAE masking")

    parser.add_argument("--dist_epochs", type=int, default=DIST_EPOCHS,
                        help="Number of distillation epochs")

    parser.add_argument("--head_epochs", type=int, default=HEAD_EPOCHS,
                        help="Number of head training epochs")

    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Override teacher checkpoint path")

    parser.add_argument("--load_backbone", type=str, default=None,
                        help="Path to distilled backbone checkpoint")

    parser.add_argument("--load_model", type=str, default=None,
                        help="Full model checkpoint for evaluation (VMamba or RETFound)")

    parser.add_argument("--dataset", type=str, default="idrid",
                        choices=["idrid", "aptos"],
                        help="Dataset selection")

    # ----------------------------------------------------------------------
    # RETFOUND-SPECIFIC ARGUMENTS
    # ----------------------------------------------------------------------
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RETFound pretrained checkpoint (optional)")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs for RETFound linear/fine-tune modes")

    args = parser.parse_args()

    # -----------------------------
    # Argument validation
    # -----------------------------
    if args.run == "head" and args.load_backbone is None:
        parser.error("--load_backbone is required for --run head")

    if args.run == "eval" and args.load_model is None:
        parser.error("--load_model is required for --run eval")

    if args.run == "retfound_eval" and args.load_model is None:
        parser.error("--load_model is required for --run retfound_eval")

    return args


def main(args):
    if args.run == "distill":
        run_distillation(args)

    elif args.run == "head":
        run_head_training(args)

    elif args.run == "eval":
        run_evaluation(args)

    # ==============================
    # RETFOUND PIPELINE (NEW)
    # ==============================
    elif args.run == "retfound_linear":
        args.retfound_mode = "linear"
        run_train_retfound(args)

    elif args.run == "retfound_finetune":
        args.retfound_mode = "finetune"
        run_train_retfound(args)

    elif args.run == "retfound_eval":
        run_eval_retfound(args)

    else:
        raise ValueError(f"Unknown mode: {args.run}")


if __name__ == "__main__":
    args = parse_args()
    main(args)