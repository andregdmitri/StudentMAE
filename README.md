# StudentMAE — README

This repository contains a unified pipeline for distilling a student VMamba model from a RETFound teacher, training classification heads, and evaluating on retinal fundus datasets (IDRiD, APTOS).

This README focuses on the CLI (`main.py`) and the actual project layout so you can run evaluation and training quickly.

---

## Repository layout (actual)

- `main.py` — unified CLI entrypoint (see CLI section below)
- `config/` — `constants.py` (defaults and environment-driven overrides)
- `dataloader/` — `idrid.py`, `aptos.py` (Lightning DataModules)
- `models/` — `retfound.py`, `vmamba_backbone.py`, `dist.py`, `models_vit.py`
- `train/` — `distill.py`, `head.py`, `train_retfound.py`
- `eval/` — `eval_vmamba.py`, `eval_retfound.py`, `shared_eval.py`
- `optimizers/` — `optimizer.py` (helper for warmup + cosine schedule)
- `ssl/` — self-supervised baselines (BYOL, Barlow Twins, SwAV)
- `utils/` — utilities (`flops.py`, `pos_embed.py`, etc.)
- `imgs/`, `results/` — example outputs and CSVs
- `requirements.txt`, `simple_test.ipynb`

Note: The directory is named `optimizers/` and contains `optimizer.py` (both names contain small typos). The codebase imports `optimizers.optimizer` accordingly. If you rename this directory/file, update all imports.

---

## CLI (`main.py`) — modes and arguments

Run `python main.py --run <mode> [options]`.

Modes (value for `--run`):

- `distill` — Phase I: distill RETFound teacher -> VMamba student
- `head` — Phase II: train classification head on distilled backbone
- `eval` — Evaluate VMamba (expects a checkpoint saved by head training)
- `retfound_linear` — RETFound linear probe (uses `train_retfound.py`)
- `retfound_finetune` — RETFound fine-tuning
- `retfound_eval` — Evaluate a RETFound Lightning checkpoint

Shared CLI arguments (most common):

- `--lr` — learning rate (default from `config/constants.py`)
- `--mask_ratio` — mask ratio used during distillation
- `--dist_epochs` — number of distillation epochs
- `--head_epochs` — number of head training epochs
- `--teacher_ckpt` — optional teacher checkpoint override
- `--load_backbone` — path to distilled backbone (required for `--run head`)
- `--load_model` — path to full model checkpoint for evaluation (required for `--run eval` and `--run retfound_eval`)
- `--dataset` — `idrid` (default) or `aptos`

RETFound-specific:

- `--checkpoint` — RETFound pretrained checkpoint (optional)
- `--epochs` — epochs for RETFound linear/finetune modes

The CLI enforces required flags per mode (e.g., `--load_backbone` for `head`). See `main.py` for the full help text.

---

## Common commands / examples

Evaluation — VMamba (IDRiD):

```bash
python main.py --run eval --load_model checkpoints/vmamba_final_head.pth --dataset idrid
```

Evaluation — VMamba (APTOS):

```bash
python main.py --run eval --load_model checkpoints/vmamba_final_head.pth --dataset aptos
```

Evaluation — RETFound Lightning ckpt:

```bash
python main.py --run retfound_eval --load_model checkpoints/retfound_finetuned.ckpt --dataset idrid
```

Distillation (Phase I):

```bash
python main.py --run distill --lr 1e-4 --mask_ratio 0.75
```

Head training (Phase II):

```bash
python main.py --run head --load_backbone checkpoints/vmamba_distilled_student.pth --lr 1e-4
```

RETFound linear / finetune:

```bash
# Linear probe
python main.py --run retfound_linear --dataset idrid --lr 3e-4

# Fine-tune
python main.py --run retfound_finetune --dataset aptos --lr 1e-5
```

Environment tips:

- Disable WandB logging: `WANDB_MODE=disabled`
- Force CPU-only runs: `CUDA_VISIBLE_DEVICES=""`

Example (disable WandB and force CPU):

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" python main.py --run eval --load_model checkpoints/vmamba_final_head.pth --dataset idrid
```

---

## Checkpoints & formats

- VMamba head output (`train/head.py`) saves a file expected by `eval` that contains two keys: `backbone` and `head` (each a state_dict).
- RETFound evaluation expects a PyTorch Lightning checkpoint compatible with `train/train_retfound.RETFoundTask`.

If you get shape mismatch errors when loading pre-trained RETFound weights, check `models/retfound.py` which contains interpolation and head-handling logic.

---

## Setup / quick start

1. Create & activate virtualenv, install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure dataset roots and checkpoint dir are set in `config/constants.py` or via environment variables (`IDRID_PATH`, `APTOS_PATH`, `CHECKPOINT_DIR`).

3. Run the desired CLI command from examples above.

---

## Known quirks & suggested small fixes

- Directory/file spelling: `optimizers/optimizer.py` contains the warmup/cosine LR helper but both names contain typos. If you prefer `optimizers/optimizer.py`, I can rename the file and update imports across the repo.
- RETFound name compatibility: code currently exposes `RETFoundBackbone`. Older notebooks may expect `RETFoundClassifier`. Adding `RETFoundClassifier = RETFoundBackbone` in `models/retfound.py` is a minimal compatibility shim.

---

If you want, I can apply either or both fixes (rename optimizer module, add the RETFound alias) and run a quick import smoke-check to validate. Tell me which action to take next.

   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   2. Configure dataset/checkpoint paths in `config/constants.py` or via environment variables:

   - `CHECKPOINT_DIR` — where trained checkpoints are saved/loaded
   - `IDRID_PATH`, `APTOS_PATH` — dataset root paths

   3. (Optional) If you use GPU, install the appropriate `torch` build. For CPU-only runs, no extra steps are needed.

   ## How to evaluate (examples)

   - VMamba evaluation (IDRiD):

   ```bash
   python main.py --run eval --load_model checkpoints/vmamba_final_head.pth --dataset idrid
   ```

   - VMamba evaluation (APTOS):

   ```bash
   python main.py --run eval --load_model checkpoints/vmamba_final_head.pth --dataset aptos
   ```

   - RETFound evaluation (expects a Lightning checkpoint for `RETFoundTask`):

   ```bash
   python main.py --run retfound_eval --load_model checkpoints/retfound_finetuned.ckpt --dataset idrid
   ```

   Environment flags you may find useful:

   - Disable WandB logging: `WANDB_MODE=disabled`
   - Force CPU (no visible GPU): `CUDA_VISIBLE_DEVICES=""`

   Example with WandB disabled on CPU:

   ```bash
   WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" python main.py --run eval --load_model checkpoints/vmamba_final_head.pth --dataset idrid
   ```

   ## How to run training (high level)

   - Phase I — Distillation (teacher RETFound -> student VMamba):

   ```bash
   python main.py --run distill --lr 1e-4 --mask_ratio 0.75
   ```

   - Phase II — Head training (requires distilled backbone):

   ```bash
   python main.py --run head --load_backbone checkpoints/vmamba_distilled_student.pth --lr 1e-4
   ```

   - RETFound workflows (linear probe / finetune):

   ```bash
   # Linear probe
   python main.py --run retfound_linear --dataset idrid --lr 3e-4

   # Fine-tune
   python main.py --run retfound_finetune --dataset aptos --lr 1e-5
   ```

   ## Notes & gotchas

   - `main.py` validates required args for each `--run` mode (e.g., `--load_model` for `--run eval`).
   - `optimizers/optimizer.py` contains the warmup-cosine helper; the filename contains a small typo (`optimizer`) but imports across the codebase use that name consistently.
   - The RETFound wrapper used in the repo exposes the class `RETFoundBackbone` — older notebooks may reference `RETFoundClassifier`. If needed, an alias can be added to `models/retfound.py` for compatibility.

   ## Troubleshooting

   - Import errors: ensure the virtualenv is activated and `requirements.txt` installed.
   - Missing checkpoints: set `CHECKPOINT_DIR` or pass full paths to `--load_model`/`--load_backbone`.

   If you want, I can:
   - rename `optimizers/optimizer.py` -> `optimizer.py` and update imports,
   - add a backwards-compatible `RETFoundClassifier` alias in `models/retfound.py`, or
   - run a quick import smoke-check to verify the CLI entrypoints load.

   ---
   Updated: current codebase (evaluation and training entrypoints available via `main.py`).