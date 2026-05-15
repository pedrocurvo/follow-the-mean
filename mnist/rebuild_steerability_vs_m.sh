#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=mnist-steerability
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=out/slurm_output/%A.out

# Avoid inheriting an already-active conda env from the submit shell.
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

source activate hfm

# Keep token in ~/.cache/huggingface/token
unset HF_HOME

# Move heavy caches off /tmp
export HF_DATASETS_CACHE=/projects/prjs1771/hf/datasets
export HF_HUB_CACHE=/projects/prjs1771/hf/hub
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Enable TF32 for cuBLAS and cuDNN
export NVIDIA_TF32_OVERRIDE=1

# Ensure async CUDA launches
export CUDA_LAUNCH_BLOCKING=0

# Reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd $HOME/scalable-fm || exit 1

PYTHON_BIN="/home/pcurvo/.conda/envs/hfm/bin/python"
srun "${PYTHON_BIN}" mnist/mnist_paper_figures.py \
  --device cuda \
  --figures steerability
