# Grokking Operators

This repository contains experiments exploring grokking behavior in small transformer models on modular arithmetic tasks. The focus is on how operator structure changes training dynamics and generalization.

## What is in this repo

- Training scripts for different operator families (e.g., addition mod p, three-term addition mod p, add-then-power mod p)
- Checkpointed runs (stored locally; not committed by default)
- Visualization utilities for loss/accuracy curves and mechanistic probes (tables, error maps, embedding PCA, cosine similarity, FFT slices, attention examples)

## Project structure

- `train_addition.py`: train on a+b mod p
- `add_three.py`: task/operator utilities for three-term addition
- `train_addpow.py`: train on (a+b)^c mod p (or related composite operators)
- `viz_addition.py`: visualization and analysis tools
- `runs/`: local experiment outputs (ignored by git by default)

## Setup

Create a virtual environment and install dependencies.

Example:

1. Create environment
2. Install requirements
3. Run a training script
4. Run visualization

## Running experiments

Addition mod p (example):

- Configure prime p, training fraction, weight decay, and model settings in `train_addition.py`
- Run training to produce a checkpoint under `runs/`
- Run `viz_addition.py` to generate plots under `runs/<run_name>/viz/`

## Notes

- Checkpoints can be large. By default, `runs/` is excluded from git.
- For reproducibility, fix random seeds and log hyperparameters alongside each run.

## License

MIT.