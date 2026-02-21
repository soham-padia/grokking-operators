#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=viz_addpow
#SBATCH --chdir=/home/padia.so/Workspace/grokking-operators
#SBATCH --output=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.out
#SBATCH --error=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/padia.so/Workspace/grokking-operators"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

mkdir -p "${PROJECT_DIR}/logs"

echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V
nvidia-smi || true
"${PYTHON_BIN}" -u "${PROJECT_DIR}/viz_addpow.py"
