#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=add_three
#SBATCH --chdir=/home/padia.so/Workspace/grokking-operators
#SBATCH --output=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.out
#SBATCH --error=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.err

set -euo pipefail
source ~/.bashrc
source /home/padia.so/Workspace/grokking-operators/.venv/bin/activate

mkdir -p logs
echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi || true
python add_three.py
