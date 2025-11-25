# main.py

import argparse
from config.constants import (
    DIST_LR,
    HEAD_LR,
    MASK_RATIO,
    DIST_EPOCHS,
    HEAD_EPOCHS,
)
from train.distill import run_distillation
from train.head import run_head_training
from train.eval import run_evaluation


def parse_args():
    epilog = """
Examples:

  # -----------------------------
  # Phase I: Distillation
  # -----------------------------
  python main.py --run distill \
      --dist_lr 1e-4 \
      --mask_ratio 0.75

  # -----------------------------
  # Phase II: Head training
  # -----------------------------
  python main.py --run head \
      --load_backbone checkpoints/vmamba_distilled_student.pth \
      --head_lr 1e-4

  # -----------------------------
  # Phase III: Evaluation
  # -----------------------------
  python main.py --run eval \
      --load_model checkpoints/vmamba_final_head.pth
"""

    parser = argparse.ArgumentParser(
        description="VMamba Training Pipeline",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # -----------------------------
    # Mode selection
    # -----------------------------
    parser.add_argument(
        "--run", type=str, default="distill",
        choices=["distill", "head", "eval"],
        help="Which phase to run: distill | head | eval"
    )

    # -----------------------------
    # Phase I distillation args
    # -----------------------------
    parser.add_argument("--dist_lr", type=float, default=DIST_LR,
                        help="Learning rate for Phase I")
    parser.add_argument("--mask_ratio", type=float, default=MASK_RATIO,
                        help="Mask ratio for student MAE masking")
    parser.add_argument("--dist_epochs", type=int, default=DIST_EPOCHS,
                        help="Number of distillation epochs")
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Override teacher checkpoint path (optional)")

    # -----------------------------
    # Phase II head training args
    # -----------------------------
    parser.add_argument("--head_lr", type=float, default=HEAD_LR,
                        help="Learning rate for Phase II")
    parser.add_argument("--head_epochs", type=int, default=HEAD_EPOCHS,
                        help="Number of head training epochs")
    parser.add_argument("--load_backbone", type=str, default=None,
                        help="Path to distilled backbone checkpoint")

    # -----------------------------
    # Phase III evaluation args
    # -----------------------------
    parser.add_argument("--load_model", type=str, default=None,
                        help="Full model checkpoint for evaluation")

    args = parser.parse_args()

    # -----------------------------
    # Argument validation
    # -----------------------------
    if args.run == "head" and args.load_backbone is None:
        parser.error("--load_backbone is required for --run head")

    if args.run == "eval" and args.load_model is None:
        parser.error("--load_model is required for --run eval")

    return args


def main(args):
    if args.run == "distill":
        run_distillation(args)

    elif args.run == "head":
        run_head_training(args)

    elif args.run == "eval":
        run_evaluation(args)

    else:
        raise ValueError(f"Unknown mode: {args.run}")


if __name__ == "__main__":
    args = parse_args()
    main(args)