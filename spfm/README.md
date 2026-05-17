# SPFM

This directory contains the image experiments for the follow-the-mean repo. It is a trimmed copy of the original SPFM training/eval code, keeping only the two model families used here:

- `spfm`: the SPFM model with a refiner.
- `dit`: the DiT model.

Removed backends include FAISS, Nystrom, BallTree, and MC variants.

## Layout

```text
spfm/
  experiments/
    spfm.yaml             # spfm experiment config
    dit.yaml      # DiT experiment config
  scripts/
    train.sh               # SLURM training launcher
    eval.sh                # SLURM evaluation launcher
  eval_tools/
    fixed_seed_comparison.py
    image_folder_metrics.py
    lpips_metrics.py
    nn_triplet.py
    nn_triplet_steering.py
  models/
    spfm/
    dit/
    refiner/
  trainer/                 # training loop helpers
  utils/                   # IO, optimizer, sampling helpers
  preprocessing/           # encoder / VAE loading
  train.py                 # main training entrypoint
  run_experiment.py        # YAML-to-CLI launcher for train.py
  eval_checkpoint.py       # checkpoint generation + metrics eval
```

## Environment

The SLURM scripts assume the Snellius-style module setup and conda env used in the original project:

```bash
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0
source activate hfm
```

The scripts use:

```bash
PYTHON_BIN=python
HF_DATASETS_CACHE=spfm/.cache/hf/datasets
HF_HUB_CACHE=spfm/.cache/hf/hub
```

Override these as environment variables if needed.

## Training

Submit the DiT config:

```bash
cd spfm
sbatch scripts/batch_train.sh experiments/dit.yaml
```

Submit the SPFM config:

```bash
cd spfm
sbatch scripts/batch_train.sh experiments/spfm.yaml
```

For a non-SLURM dry run that only prints the generated command:

```bash
python run_experiment.py --config experiments/spfm.yaml --dry-run
python run_experiment.py --config experiments/dit.yaml --dry-run
```

Outputs default to:

```text
out/spfm_afhq_cat_dog
out/dit_afhq_10k
```

## Experiment Configs

`experiments/spfm.yaml` trains `spfm` with:

- AFHQv2 train split.
- Main DB `dog:1.0,cat:1.0`.
- Alt DB `dog:0.0,cat:1.0`.
- Refiner depth `11`.
- Output directory `out/spfm_afhq_cat_dog`.

`experiments/dit.yaml` trains `dit` with:

- AFHQv2 train split.
- Main DB `dog:1.0,cat:1.0`.
- Alt DB `dog:0.3,cat:0.7`.
- Refiner depth `0`.
- Output directory `out/dit_afhq_10k`.

## Evaluation

Use one launcher for all eval modes:

```bash
cd spfm
MODE=spfm-catdog sbatch --gres=gpu:4 scripts/batch_eval.sh
```

Common overrides:

```bash
CONFIG=experiments/spfm.yaml
CKPT=out/spfm_afhq_cat_dog/model_step10000.pt
MODEL_TAG=my_eval_name
RESULTS_DIR=out/evals/custom
ARTIFACTS_ROOT=out/evals/artifacts/my_eval_name
TOTAL_GEN=50000
SAMPLE_STEPS=20
GEN_BATCH=128
USE_EMA=false
NUM_PROCESSES=4
```

Supported modes:

```bash
MODE=spfm-catdog sbatch --gres=gpu:4 scripts/batch_eval.sh
MODE=dit-catdog sbatch --gres=gpu:4 scripts/batch_eval.sh
MODE=db-size-sweep DB_SIZES="10 100 1000" sbatch --gres=gpu:1 scripts/batch_eval.sh
MODE=class-balance CAT_PCTS="100 50 0" TOTAL_GEN=1000 sbatch --gres=gpu:1 scripts/batch_eval.sh
MODE=lpips DB_SIZES="10 100 1000" COMPOSITIONS="cat100_dog0 cat50_dog50" sbatch --gres=gpu:1 scripts/batch_eval.sh
MODE=nn-triplet sbatch --gres=gpu:1 scripts/batch_eval.sh
MODE=nn-triplet-steer STEER_STRENGTH=1.0 sbatch --gres=gpu:1 scripts/batch_eval.sh
MODE=metrics-only GENERATED_DIR=out/generated REFERENCE_DIR=out/reference sbatch --gres=gpu:1 scripts/batch_eval.sh
MODE=fixed-seed sbatch --gres=gpu:1 scripts/batch_eval.sh
```

Evaluation outputs go under `out/evals/...` by default. Generated image artifacts go under `ARTIFACTS_ROOT` so large files do not have to live in the repo.

## Direct Python Tools

Most SLURM runs should go through the local `scripts/batch_eval.sh` wrapper. The core
`scripts/eval.sh` and helper tools can also be called directly from an already-prepared
environment:

```bash
python eval_checkpoint.py --help
python eval_tools/image_folder_metrics.py --help
python eval_tools/nn_triplet.py --help
python eval_tools/nn_triplet_steering.py --help
python eval_tools/lpips_metrics.py --help
python eval_tools/fixed_seed_comparison.py --help
```

## Notes

- `scripts/batch_train.sh` and `scripts/batch_eval.sh` contain local SLURM/module setup and are ignored by git.
- `scripts/train.sh` and `scripts/eval.sh` resolve `spfm/` relative to their own location and contain the reusable run logic.
- `run_experiment.py` flattens YAML config sections into `train.py` CLI flags.
- `eval_checkpoint.py` loads the same YAML config, restores model weights, generates samples, and computes metrics.
- Checkpoints can be passed as `model_step*.pt`, `model_last.pt`, or a checkpoint directory when supported by the eval tool.
