<h2 align="center">Reference-Guided Flow Matching</h2>

<p align="center">
  <img src="assets/follow-the-mean.gif" alt="Follow the Mean reference-guided generation" width="680">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.10302"><img src="https://img.shields.io/badge/arXiv-2605.10302-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pedrocurvo.com/follow-the-mean"><img src="https://img.shields.io/badge/Blog-follow--the--mean-1f6feb.svg" alt="blog"></a>
  <img src="https://img.shields.io/badge/method-reference--guided--flows-4b44ce.svg" alt="method">
  <img src="https://img.shields.io/badge/control-examples--not--rewards-f9c74f.svg" alt="examples not rewards">
</p>

This repository contains the experiments for **Follow the Mean: Reference-Guided Flow Matching**.

The core idea is simple: in flow matching, the velocity field is governed by an endpoint mean. If you shift that endpoint mean, you shift the flow. This turns controllable generation into a data problem: choose a reference set that expresses the attribute, object, style, pose, or target distribution you want, and let the sampler follow the mean induced by those examples.

In short:

```text
guide with examples, not rewards
```

The repository walks through that idea in four stages:

1. `moons/` makes the mechanism visible in two dimensions.
2. `mnist/` shows the same sparse-reference idea on simple images.
3. `rgm/` applies Reference-Mean Guidance to a frozen FLUX.2-klein text-to-image model.
4. `spfm/` contains the semi-parametric image-model experiments that amortize reference-set guidance into a learned architecture.

No single folder tells the whole story. The point of the repo is the progression: start where every vector field can be drawn, then move toward real image generation.

## The Story

Most control methods for generative models ask for one of three things:

- update model parameters through fine-tuning or adapters,
- train or call an auxiliary classifier or reward model,
- search over many candidates at inference time.

Reference-guided flow matching asks for something different: examples.

Suppose a flow model is generating from a noisy state `x_t`. For deterministic interpolants, the velocity points toward an endpoint mean. If the original model follows the mean of its training distribution, then a controlled model can be seen as following a different endpoint mean.

RMG approximates that target mean from a reference set:

```text
R = {reference image 1, reference image 2, ..., reference image M}
```

At sampling time, the method computes an empirical reference endpoint mean and adds a guidance velocity:

```text
v_guided(x_t, t) = (mu_ref(x_t, t) - x_t) / (1 - t)
```

The practical consequence is important: the control signal lives in the reference set, not in a trained classifier, reward, prompt optimizer, or model update. Change the references, and the flow changes.

## Repository Map

```text
follow-the-mean-rgfm/
  moons/      # Fully visualizable two-dimensional mechanism
  mnist/      # Sparse-reference guidance on MNIST zeros and ones
  rgm/        # Training-free RMG on frozen FLUX.2-klein
  spfm/       # Semi-Parametric Guidance training and evaluation code
```

Each subfolder has its own README with more detail and exact commands.

## Experiments

### Two Moons

`moons/` is the smallest experiment. It uses a two-dimensional dataset so the posterior field, velocity field, trajectories, and generated samples can all be drawn.

It answers:

- Can a few labeled reference points induce a useful posterior field?
- Does following that soft posterior move samples toward the target class?
- Does changing the reference composition change the generated distribution?
- How many references are needed before soft guidance approaches a hard-label oracle?

Run it with:

```bash
cd moons
python figures.py
```

Expected outputs are written to `moons/images/`.

### MNIST

`mnist/` repeats the same idea on image vectors. It keeps only digits `0` and `1`, uses a sparse balanced reference set, and asks whether the inferred labels can guide generation toward the requested digit.

It answers:

- At what flow time do sparse labels become useful?
- How does accuracy improve with the number of references?
- Can sparse references generate recognizable zeros and ones?
- How close does soft sparse-reference guidance get to a hard-label oracle?

Run all MNIST figures with:

```bash
cd mnist
python figures.py --device auto --figures all
```

The expensive steerability sweep can also be submitted through:

```bash
cd mnist
sbatch run_steerability.sh
```

### RMG With FLUX.2

`rgm/` is the training-free pretrained-model endpoint. It applies Reference-Mean Guidance to a frozen `black-forest-labs/FLUX.2-klein-4B` model.

The prompt, seed, sampler, and model weights can stay fixed while the reference set changes. This is used for:

- color control, such as elephant -> pink elephant or blue elephant,
- object identity, such as animal -> giraffe or zebra,
- style control, such as cat -> Van Gogh or studio photograph,
- structural/image-reference control, such as hand poses or ring-leap poses,
- reference composition sweeps,
- guidance schedule and strength ablations,
- reference-size and sampling-step ablations.

Run a config with:

```bash
cd rgm
./run.sh experiments/pink_elephant.yaml
```

Dry-run a config without launching generation:

```bash
cd rgm
./run.sh experiments/pink_elephant.yaml --dry-run
```

See `rgm/README.md` for the full YAML schema, output layout, evaluation helpers, and reference-cache notes.

### SPFM / SPG

`spfm/` contains the learned counterpart: Semi-Parametric Guidance (SPG). Instead of computing the reference mean only at inference time, SPG amortizes the same idea into a model with an explicit reference-set anchor and learned residual refiner.

This folder keeps the image-model training and evaluation code for:

- `spfm`: the reference-conditioned semi-parametric model,
- `dit`: the unconditional DiT baseline.

Submit training through:

```bash
cd spfm
sbatch scripts/train.sh experiments/spfm.yaml
sbatch scripts/train.sh experiments/dit.yaml
```

Dry-run the generated training command with:

```bash
cd spfm
python run_experiment.py --config experiments/spfm.yaml --dry-run
```

See `spfm/README.md` for environment assumptions, training, evaluation modes, and checkpoint tooling.

## Setup

The lightweight experiments (`moons/`, `mnist/`) use standard Python scientific packages. The image experiments require a heavier environment with PyTorch, diffusers, transformers, and GPU access.

The heavier image folders keep separate `pyproject.toml` files because their environments are not identical:

```bash
cd rgm
python -m pip install -e ".[eval]"

cd ../spfm
python -m pip install -e ".[metrics]"
```

A typical environment needs:

```text
python
torch
numpy
scipy
matplotlib
scikit-learn
torchvision
diffusers
transformers
accelerate
Pillow
```

For RMG with FLUX.2, you also need access to:

```text
black-forest-labs/FLUX.2-klein-4B
```

On Snellius-style systems, the scripts assume the project environment used for the experiments. See `spfm/README.md` and `rgm/README.md` for folder-specific notes.

## Outputs And Git Hygiene

Generated artifacts are intentionally ignored by git:

- figures and image outputs,
- checkpoints,
- tensor caches,
- RMG reference caches under `rgm/references/`,
- experiment runs under `runs/`, `out/`, or `outputs/`.

The repository keeps only source code, configs, lightweight published figures, and placeholder directories such as `rgm/references/.gitkeep`.

## Paper

```bibtex
@misc{curvo2026followmean,
  title        = {Follow the Mean: Reference-Guided Flow Matching},
  author       = {Pedro M. P. Curvo and Maksim Zhdanov and Floor Eijkelboom and Jan-Willem van de Meent},
  year         = {2026},
  eprint       = {2605.10302},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

## Why This Repo Matters

The main claim is not that every target can be solved by a small reference set. The claim is more specific and more useful: flow matching exposes a natural control interface through endpoint means.

The toy experiments show the mechanism. MNIST shows it survives a jump to image vectors. RMG shows it can control a frozen pretrained image generator without new weights or reward models. SPG shows the same principle can be built into a learnable architecture.

Together, the folders support one throughline:

```text
change the reference set -> change the endpoint mean -> change the flow
```
