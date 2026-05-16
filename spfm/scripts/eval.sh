#!/bin/bash
# Unified SPFM evaluation launcher.
#
# Submit with one MODE and override settings through environment variables:
#
#   sbatch --gres=gpu:4 scripts/eval.sh
#   MODE=dit-catdog sbatch --gres=gpu:4 scripts/eval.sh
#   MODE=db-size-sweep DB_SIZES="10 100 1000" sbatch --gres=gpu:1 scripts/eval.sh
#   MODE=class-balance CAT_PCTS="100 50 0" TOTAL_GEN=1000 sbatch --gres=gpu:1 scripts/eval.sh
#   MODE=lpips DB_SIZES="10 100 1000" COMPOSITIONS="cat100_dog0 cat50_dog50" sbatch --gres=gpu:1 scripts/eval.sh
#   MODE=nn-triplet sbatch --gres=gpu:1 scripts/eval.sh
#   MODE=nn-triplet-steer STEER_STRENGTH=1.0 sbatch --gres=gpu:1 scripts/eval.sh
#   MODE=metrics-only GENERATED_DIR=out/generated REFERENCE_DIR=out/reference sbatch --gres=gpu:1 scripts/eval.sh
#   MODE=fixed-seed sbatch --gres=gpu:1 scripts/eval.sh
#
# Common overrides:
#   CONFIG=experiments/spfm.yaml
#   CKPT=out/spfm_afhq_cat_dog/model_step10000.pt
#   MODEL_TAG=my_eval_name
#   RESULTS_DIR=out/evals/custom
#   ARTIFACTS_ROOT=out/evals/artifacts/my_eval_name
#   TOTAL_GEN=50000 SAMPLE_STEPS=20 GEN_BATCH=128 USE_EMA=false NUM_PROCESSES=4
#
# Modes:
#   spfm-catdog       SPFM cat/dog 50k eval with FID/KID/IS/CLIP.
#   dit-catdog   DiT cat/dog 50k eval with FID/KID/IS/CLIP.
#   db-size-sweep     SPFM sweep over N_img values.
#   class-balance     SPFM sweep over cat/dog DB composition.
#   lpips             SPFM LPIPS sweep over DB sizes/compositions.
#   nn-triplet        SPFM 3-way nearest-neighbor visualization.
#   nn-triplet-steer  SPFM 3-way steering visualization.
#   metrics-only      Recompute metrics for existing generated/reference dirs.
#   fixed-seed        Fixed-seed DiT vs SPFM comparison.
#
# This file replaces the old one-off eval_*.sh wrappers. The Python helpers in
# eval_tools/ contain model-specific evaluation logic.

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=spfm-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=out/slurm_output/%A.out

set -euo pipefail

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0

source activate hfm

unset HF_HOME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPFM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SPFM_DIR}" || exit 1

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SPFM_DIR}/.cache/hf/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SPFM_DIR}/.cache/hf/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-spfm-catdog}"
CONFIG="${CONFIG:-}"
MODEL_DIR="${MODEL_DIR:-}"
CKPT="${CKPT:-}"
MODEL_TAG="${MODEL_TAG:-}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-}"
TOTAL_GEN="${TOTAL_GEN:-50000}"
SAMPLE_STEPS="${SAMPLE_STEPS:-20}"
GEN_BATCH="${GEN_BATCH:-128}"
DECODE_BATCH="${DECODE_BATCH:-16}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
USE_EMA="${USE_EMA:-false}"

run_eval_checkpoint() {
  local maybe_multi=()
  if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    maybe_multi=(-m accelerate.commands.launch --multi_gpu --num_processes "${NUM_PROCESSES}")
  fi
  srun "${PYTHON_BIN}" "${maybe_multi[@]}" eval_checkpoint.py "$@"
}

set_spfm_defaults() {
  CONFIG="${CONFIG:-experiments/spfm.yaml}"
  MODEL_DIR="${MODEL_DIR:-out/spfm_afhq_cat_dog}"
  CKPT="${CKPT:-${MODEL_DIR}/model_step10000.pt}"
  MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_DIR}")_step10000}"
  ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-out/evals/artifacts/${MODEL_TAG}}"
}

set_dit_defaults() {
  CONFIG="${CONFIG:-experiments/dit.yaml}"
  MODEL_DIR="${MODEL_DIR:-out/dit_afhq_10k}"
  CKPT="${CKPT:-${MODEL_DIR}/model_step10000.pt}"
  MODEL_TAG="${MODEL_TAG:-dit_afhq_10k_step10000}"
  ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-out/evals/artifacts/${MODEL_TAG}}"
}

run_model_catdog() {
  local default_results="out/evals/${MODEL_TAG}/catdog_50k"
  local results_dir="${RESULTS_DIR:-${default_results}}"
  local generated_dir="${GENERATED_DIR:-${ARTIFACTS_ROOT}/catdog_50k/generated}"
  local reference_dir="${REFERENCE_DIR:-${ARTIFACTS_ROOT}/catdog_50k/reference_db}"

  echo "[eval.sh] mode=${MODE} config=${CONFIG} ckpt=${CKPT}"
  echo "[eval.sh] results_dir=${results_dir} generated_dir=${generated_dir}"
  run_eval_checkpoint \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --use_ema "${USE_EMA}" \
    --results_dir "${results_dir}" \
    --generated_dir "${generated_dir}" \
    --reference_dir "${reference_dir}" \
    --total_gen "${TOTAL_GEN}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --gen_batch "${GEN_BATCH}" \
    --db_override "dog:1.0,cat:1.0" \
    --alt_db_override none \
    --reference_mode db \
    --fid true \
    --kid true \
    --inception_score true \
    --clip_eval true \
    --clip_eval_prompts "a photo of a dog" \
    --clip_eval_prompts "a photo of a cat"
}

run_db_size_sweep() {
  local results_root="${RESULTS_ROOT:-out/evals/${MODEL_TAG}/db_size_sweep}"
  local reference_dir="${REFERENCE_DIR:-${ARTIFACTS_ROOT}/reference/afhq_train}"
  local db_sizes=(${DB_SIZES:-10 100 200 500 1000 2000 5000 10000 15000})

  for db_size in "${db_sizes[@]}"; do
    echo "[eval.sh] db-size-sweep N_img=${db_size}"
    run_eval_checkpoint \
      --config "${CONFIG}" \
      --ckpt "${CKPT}" \
      --use_ema "${USE_EMA}" \
      --results_dir "${results_root}/N${db_size}" \
      --generated_dir "${ARTIFACTS_ROOT}/db_size_sweep/N${db_size}/generated" \
      --reference_dir "${reference_dir}" \
      --total_gen "${TOTAL_GEN}" \
      --sample_steps "${SAMPLE_STEPS}" \
      --gen_batch "${GEN_BATCH}" \
      --n_img_override "${db_size}" \
      --db_override none \
      --alt_db_override none \
      --reference_mode dataset \
      --fid true \
      --kid true \
      --inception_score true \
      --clip_eval true
  done
}

run_class_balance() {
  local results_root="${RESULTS_ROOT:-out/evals/${MODEL_TAG}/class_balance_sweep}"
  local cat_pcts=(${CAT_PCTS:-100 90 80 70 60 50 40 30 20 10 0})

  for cat_pct in "${cat_pcts[@]}"; do
    local dog_pct=$((100 - cat_pct))
    local db_spec
    if [[ -n "${SWEEP_N_IMG:-}" ]]; then
      local cat_count=$(( (SWEEP_N_IMG * cat_pct + 50) / 100 ))
      local dog_count=$(( SWEEP_N_IMG - cat_count ))
      db_spec="cat=${cat_count},dog=${dog_count}"
    else
      db_spec="cat:$(awk "BEGIN {printf \"%.1f\", ${cat_pct}/100}"),dog:$(awk "BEGIN {printf \"%.1f\", ${dog_pct}/100}")"
    fi
    echo "[eval.sh] class-balance db=${db_spec}"
    run_eval_checkpoint \
      --config "${CONFIG}" \
      --ckpt "${CKPT}" \
      --use_ema "${USE_EMA}" \
      --results_dir "${results_root}/cat${cat_pct}_dog${dog_pct}" \
      --generated_dir "${ARTIFACTS_ROOT}/class_balance_sweep/cat${cat_pct}_dog${dog_pct}/generated" \
      --total_gen "${TOTAL_GEN}" \
      --sample_steps "${SAMPLE_STEPS}" \
      --gen_batch "${GEN_BATCH}" \
      --db_override "${db_spec}" \
      --alt_db_override none \
      --reference_mode none \
      --fid false \
      --kid false \
      --inception_score false \
      --clip_eval true \
      --clip_eval_prompts "a photo of a dog" \
      --clip_eval_prompts "a photo of a cat"
  done
}

run_lpips_sweep() {
  local results_root="${RESULTS_ROOT:-out/evals/${MODEL_TAG}/lpips_sweep}"
  local db_sizes=(${DB_SIZES:-10 50 100 200 500 1000 5000})
  local compositions=(${COMPOSITIONS:-cat100_dog0 cat50_dog50 cat0_dog100})
  local lpips_max_pairs="${LPIPS_MAX_PAIRS:-0}"
  local lpips_batch_size="${LPIPS_BATCH_SIZE:-64}"

  for n_img in "${db_sizes[@]}"; do
    for comp in "${compositions[@]}"; do
      local cat_pct dog_pct
      case "${comp}" in
        cat100_dog0) cat_pct=100; dog_pct=0 ;;
        cat50_dog50) cat_pct=50; dog_pct=50 ;;
        cat0_dog100) cat_pct=0; dog_pct=100 ;;
        *) echo "Unsupported composition: ${comp}" >&2; exit 1 ;;
      esac
      local cat_count=$(( (n_img * cat_pct + 50) / 100 ))
      local dog_count=$(( n_img - cat_count ))
      local db_spec="cat=${cat_count},dog=${dog_count}"
      local variant="N${n_img}_${comp}"
      local results_dir="${results_root}/${variant}"
      local generated_dir="${ARTIFACTS_ROOT}/lpips_sweep/${variant}/generated"

      echo "[eval.sh] lpips variant=${variant} db=${db_spec}"
      run_eval_checkpoint \
        --config "${CONFIG}" \
        --ckpt "${CKPT}" \
        --use_ema "${USE_EMA}" \
        --results_dir "${results_dir}" \
        --generated_dir "${generated_dir}" \
        --total_gen "${TOTAL_GEN}" \
        --sample_steps "${SAMPLE_STEPS}" \
        --gen_batch "${GEN_BATCH}" \
        --db_override "${db_spec}" \
        --alt_db_override none \
        --reference_mode none \
        --fid false \
        --kid false \
        --inception_score false \
        --clip_eval false

      srun "${PYTHON_BIN}" eval_tools/lpips_metrics.py \
        --image_dir "${generated_dir}" \
        --output_json "${results_dir}/lpips_metrics.json" \
        --max_pairs "${lpips_max_pairs}" \
        --batch_size "${lpips_batch_size}" \
        --seed 1234
    done
  done
}

run_nn_triplet() {
  local results_dir="${RESULTS_DIR:-out/evals/${MODEL_TAG}/nn_triplet}"
  srun "${PYTHON_BIN}" eval_tools/nn_triplet.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --results_dir "${results_dir}" \
    --num_gen "${NUM_GEN:-8}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --decode_batch "${DECODE_BATCH}" \
    --use_ema "${USE_EMA}" \
    --seed "${SEED:-1234}"
}

run_nn_triplet_steer() {
  local results_dir="${RESULTS_DIR:-out/evals/${MODEL_TAG}/nn_triplet_steer}"
  local extra_args=()
  [[ -n "${PATCH_ROWS:-}" ]] && extra_args+=(--patch_rows "${PATCH_ROWS}")
  [[ -n "${PATCH_COLS:-}" ]] && extra_args+=(--patch_cols "${PATCH_COLS}")

  srun "${PYTHON_BIN}" eval_tools/nn_triplet_steering.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --results_dir "${results_dir}" \
    --num_gen "${NUM_GEN:-8}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --decode_batch "${DECODE_BATCH}" \
    --use_ema "${USE_EMA}" \
    --seed "${SEED:-1234}" \
    --steer_strength "${STEER_STRENGTH:-1.0}" \
    --beta_schedule "${BETA_SCHEDULE:-quadratic-decay}" \
    --steer_start_frac "${STEER_START_FRAC:-0.15}" \
    --steer_end_frac "${STEER_END_FRAC:-0.95}" \
    --steer_topk "${STEER_TOPK:-0}" \
    --class_subset_size "${CLASS_SUBSET_SIZE:-0}" \
    --class_subset_mode "${CLASS_SUBSET_MODE:-white_background}" \
    "${extra_args[@]}"
}

run_metrics_only() {
  if [[ -z "${GENERATED_DIR:-}" || -z "${REFERENCE_DIR:-}" ]]; then
    echo "MODE=metrics-only requires GENERATED_DIR and REFERENCE_DIR." >&2
    exit 1
  fi
  srun "${PYTHON_BIN}" eval_tools/image_folder_metrics.py \
    --generated_dir "${GENERATED_DIR}" \
    --reference_dir "${REFERENCE_DIR}" \
    --results_dir "${RESULTS_DIR:-out/evals/${MODEL_TAG}/metrics_only}" \
    --config "${CONFIG}" \
    --fid "${FID:-true}" \
    --kid "${KID:-true}" \
    --inception_score "${INCEPTION_SCORE:-true}" \
    --device "${DEVICE:-cuda}"
}

run_fixed_seed() {
  srun "${PYTHON_BIN}" eval_tools/fixed_seed_comparison.py \
    --config "${CONFIG:-experiments/spfm.yaml}" \
    --dit-ckpt "${DIT_CKPT:-out/dit_afhq_10k}" \
    --spfm-ckpt "${SPFM_CKPT:-${CKPT}}" \
    --output-root "${OUTPUT_ROOT:-out/evals/fixed_seed_comparison}" \
    --seed "${SEED:-1234}" \
    --sample-steps "${SAMPLE_STEPS}" \
    --decode-batch "${DECODE_BATCH}" \
    --use-ema "${USE_EMA}" \
    --cats "${CATS:-50}" \
    --dogs "${DOGS:-50}" \
    --steer-strength "${STEER_STRENGTH:-1.0}" \
    --beta-schedule "${BETA_SCHEDULE:-quadratic-decay}" \
    --steer-start-frac "${STEER_START_FRAC:-0.15}" \
    --steer-end-frac "${STEER_END_FRAC:-0.95}" \
    --steer-topk "${STEER_TOPK:-0}" \
    --device "${DEVICE:-cuda}"
}

case "${MODE}" in
  spfm-catdog)
    set_spfm_defaults
    run_model_catdog
    ;;
  dit-catdog)
    set_dit_defaults
    run_model_catdog
    ;;
  db-size-sweep)
    set_spfm_defaults
    run_db_size_sweep
    ;;
  class-balance)
    set_spfm_defaults
    run_class_balance
    ;;
  lpips)
    set_spfm_defaults
    run_lpips_sweep
    ;;
  nn-triplet)
    set_spfm_defaults
    run_nn_triplet
    ;;
  nn-triplet-steer)
    set_spfm_defaults
    run_nn_triplet_steer
    ;;
  metrics-only)
    set_spfm_defaults
    run_metrics_only
    ;;
  fixed-seed)
    set_spfm_defaults
    run_fixed_seed
    ;;
  *)
    echo "Unknown MODE='${MODE}'. Open scripts/eval.sh for supported modes and examples." >&2
    exit 2
    ;;
esac
