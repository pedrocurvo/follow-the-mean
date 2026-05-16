# MNIST Steering Figures

This folder contains the closed-form MNIST experiment used to visualize sparse-label steering on digits `0` and `1`.

The script does not train a neural network. It loads MNIST, builds a small labeled set, estimates label posteriors with the flow kernel, and generates figures showing how well sparse labels steer samples toward the target digit.

## Files

```text
mnist/
  figures.py              # Main figure-generation CLI
  run_steerability.sh     # SLURM helper for the expensive steerability figure
  images/                 # Generated figures; ignored by git
```

## Figures

`figures.py` can generate:

- `accuracy-t`: label propagation accuracy as a function of flow time `t`.
- `accuracy-m`: label propagation accuracy as the number of labeled examples `M` changes.
- `samples`: generated zero/one sample grids.
- `steerability`: target-digit generation rate as `M` changes.
- `all`: every figure above.

## Run Locally

From this folder:

```bash
cd mnist
python figures.py --device auto --figures all
```

Generate only the cached steerability plot:

```bash
python figures.py --device auto --figures steerability
```

Useful overrides:

```bash
python figures.py \
  --data-root data/mnist \
  --output-dir images \
  --n-mnist 1000 \
  --m-values 2 5 10 20 50 \
  --n-trials 5 \
  --figures accuracy-t accuracy-m samples
```

## Run On SLURM

The steerability sweep is the slowest path. Submit it with:

```bash
cd mnist
sbatch run_steerability.sh
```

## Outputs

Generated files are written to `images/`:

```text
mnist_accuracy_vs_t.pdf
mnist_accuracy_vs_m.pdf
mnist_generated_zeros.pdf
mnist_generated_ones.pdf
mnist_steerability_vs_m.pdf
mnist_steerability_vs_m.npz
```

The `.npz` file caches the steerability sweep so the plot can be rebuilt without rerunning the expensive generation loop.
