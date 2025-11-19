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
    parser = argparse.ArgumentParser(description="VMamba Training Pipeline")

    parser.add_argument("--run", type=str, default="distill",
                        choices=["distill", "head", "eval"])

    parser.add_argument("--dist_lr", type=float, default=DIST_LR,
                        help="Learning rate for Phase I distillation")
    parser.add_argument("--head_lr", type=float, default=HEAD_LR,
                        help="Learning rate for Phase II head training")

    parser.add_argument("--mask_ratio", type=float, default=MASK_RATIO)

    parser.add_argument("--dist_epochs", type=int, default=DIST_EPOCHS)
    parser.add_argument("--head_epochs", type=int, default=HEAD_EPOCHS)

    parser.add_argument("--load_backbone", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)

    return parser.parse_args()


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
