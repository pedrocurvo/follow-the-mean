# SPFM

This directory contains the image experiments for the follow-the-mean repo. It is a trimmed copy of the original SPFM training/eval code, keeping only the two model families used here:

- `full_attention`: the retrieval model with a refiner.
- `baseline_dit`: the baseline DiT model.

Removed backends include FAISS, Nystrom, BallTree, and MC variants.

## Layout

```text
spfm/
  experiments/
    model.yaml             # full_attention experiment config
    baseline_dit.yaml      # baseline DiT experiment config
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
    full_attention/
    baseline_dit/
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
PYTHON_BIN=/home/pcurvo/.conda/envs/hfm/bin/python
HF_DATASETS_CACHE=/projects/prjs1771/hf/datasets
HF_HUB_CACHE=/projects/prjs1771/hf/hub
```

Override these as environment variables if needed.

## Training

Submit the baseline DiT config:

```bash
cd /home/pcurvo/follow-the-mean-rgfm/spfm
sbatch scripts/train.sh experiments/baseline_dit.yaml
```

Submit the full-attention config:

```bash
cd /home/pcurvo/follow-the-mean-rgfm/spfm
sbatch scripts/train.sh experiments/model.yaml
```

For a non-SLURM dry run that only prints the generated command:

```bash
/home/pcurvo/.conda/envs/hfm/bin/python run_experiment.py --config experiments/model.yaml --dry-run
/home/pcurvo/.conda/envs/hfm/bin/python run_experiment.py --config experiments/baseline_dit.yaml --dry-run
```

Outputs default to:

```text
out/model_spf_fullattention_afhq_cat_dog
out/model_baseline_dit_afhq_10k
```

## Experiment Configs

`experiments/model.yaml` trains `full_attention` with:

- AFHQv2 train split.
- Main DB `dog:1.0,cat:1.0`.
- Alt DB `dog:0.0,cat:1.0`.
- Refiner depth `11`.
- Output directory `out/model_spf_fullattention_afhq_cat_dog`.

`experiments/baseline_dit.yaml` trains `baseline-dit` with:

- AFHQv2 train split.
- Main DB `dog:1.0,cat:1.0`.
- Alt DB `dog:0.3,cat:0.7`.
- Refiner depth `0`.
- Output directory `out/model_baseline_dit_afhq_10k`.

## Evaluation

Use one launcher for all eval modes:

```bash
cd /home/pcurvo/follow-the-mean-rgfm/spfm
MODE=full-catdog sbatch --gres=gpu:4 scripts/eval.sh
```

Common overrides:

```bash
CONFIG=experiments/model.yaml
CKPT=out/model_spf_fullattention_afhq_cat_dog/model_step10000.pt
MODEL_TAG=my_eval_name
RESULTS_DIR=out/evals/custom
ARTIFACTS_ROOT=/projects/prjs1771/follow-the-mean-rgfm/evals/my_eval_name
TOTAL_GEN=50000
SAMPLE_STEPS=20
GEN_BATCH=128
USE_EMA=false
NUM_PROCESSES=4
```

Supported modes:

```bash
MODE=full-catdog sbatch --gres=gpu:4 scripts/eval.sh
MODE=baseline-catdog sbatch --gres=gpu:4 scripts/eval.sh
MODE=db-size-sweep DB_SIZES="10 100 1000" sbatch --gres=gpu:1 scripts/eval.sh
MODE=class-balance CAT_PCTS="100 50 0" TOTAL_GEN=1000 sbatch --gres=gpu:1 scripts/eval.sh
MODE=lpips DB_SIZES="10 100 1000" COMPOSITIONS="cat100_dog0 cat50_dog50" sbatch --gres=gpu:1 scripts/eval.sh
MODE=nn-triplet sbatch --gres=gpu:1 scripts/eval.sh
MODE=nn-triplet-steer STEER_STRENGTH=1.0 sbatch --gres=gpu:1 scripts/eval.sh
MODE=metrics-only GENERATED_DIR=/path/generated REFERENCE_DIR=/path/reference sbatch --gres=gpu:1 scripts/eval.sh
MODE=fixed-seed sbatch --gres=gpu:1 scripts/eval.sh
```

Evaluation outputs go under `out/evals/...` by default. Generated image artifacts go under `ARTIFACTS_ROOT` so large files do not have to live in the repo.

## Direct Python Tools

Most runs should go through `scripts/eval.sh`, but the helper tools can be called directly:

```bash
/home/pcurvo/.conda/envs/hfm/bin/python eval_checkpoint.py --help
/home/pcurvo/.conda/envs/hfm/bin/python eval_tools/image_folder_metrics.py --help
/home/pcurvo/.conda/envs/hfm/bin/python eval_tools/nn_triplet.py --help
/home/pcurvo/.conda/envs/hfm/bin/python eval_tools/nn_triplet_steering.py --help
/home/pcurvo/.conda/envs/hfm/bin/python eval_tools/lpips_metrics.py --help
/home/pcurvo/.conda/envs/hfm/bin/python eval_tools/fixed_seed_comparison.py --help
```

## Notes

- `scripts/train.sh` and `scripts/eval.sh` resolve `spfm/` relative to their own location, so they can be submitted from inside `spfm/` or through an absolute path.
- `run_experiment.py` flattens YAML config sections into `train.py` CLI flags.
- `eval_checkpoint.py` loads the same YAML config, restores model weights, generates samples, and computes metrics.
- Checkpoints can be passed as `model_step*.pt`, `model_last.pt`, or a checkpoint directory when supported by the eval tool.
