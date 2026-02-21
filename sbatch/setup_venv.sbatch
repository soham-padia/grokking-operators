#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=setup_venv
#SBATCH --chdir=/home/padia.so/Workspace/grokking-operators
#SBATCH --output=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.out
#SBATCH --error=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/padia.so/Workspace/grokking-operators"
VENV_DIR="${PROJECT_DIR}/.venv"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
LOG_DIR="${PROJECT_DIR}/logs"
DONE_FILE="${LOG_DIR}/setup_venv_done_${SLURM_JOB_ID}.txt"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

echo "Setup complete at $(date)" | tee "${DONE_FILE}"
echo "Done marker: ${DONE_FILE}"
