# RGM Flux Experiments

This directory contains the curated Flux RMG experiments copied out of `scalable-fm`.

- `references/`: cached/generated reference sets needed by the kept experiments.
- `experiments/`: one YAML config per experiment. Single generations use `case.reference_set`; sweeps use dedicated sections.
- `src/`: Python experiment code and shared YAML/runtime helpers.
- `scripts/run_experiment.py`: YAML launcher that validates configs, stages references, and runs the selected Python file with `--config`.
- `run.sh`: convenience entry point.

Run a config with:

```bash
cd /projects/prjs1771/follow-the-mean-rgfm/rgm
./run.sh experiments/pink_elephant.yaml
```

Use `--dry-run` to print the resolved command without launching generation.

Direct script usage is also config-driven:

```bash
python src/training_free_control.py --config experiments/pink_elephant.yaml
```

Single guided generations use the same schema for text and image reference sets:

```yaml
case:
  slug: pink_elephant
  prompt: an elephant in a jungle
  reference_set:
    type: prompts
    label: pink elephant reference
    prompt: a pink elephant
```
