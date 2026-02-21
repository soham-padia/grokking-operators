#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --job-name=train_addpow_loop
#SBATCH --chdir=/home/padia.so/Workspace/grokking-operators
#SBATCH --output=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.out
#SBATCH --error=/home/padia.so/Workspace/grokking-operators/logs/%x-%j.err

set -uo pipefail

PROJECT_DIR="/home/padia.so/Workspace/grokking-operators"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
SELF_SCRIPT="${PROJECT_DIR}/sbatch/train_addpow_autoloop.sbatch"
SOLVED_FILE="${PROJECT_DIR}/runs/addpow_p97_cmax32_tf0.2_wd0.1/SOLVED_DONE"
MAX_FILE="${PROJECT_DIR}/runs/addpow_p97_cmax32_tf0.2_wd0.1/MAX_STEPS_REACHED"

mkdir -p "${PROJECT_DIR}/logs"

if [ -f "${SOLVED_FILE}" ]; then
  echo "Solved marker already present: ${SOLVED_FILE}"
  exit 0
fi
if [ -f "${MAX_FILE}" ]; then
  echo "Hard max marker already present: ${MAX_FILE}"
  exit 0
fi

NEXT_JOB_ID=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "${SELF_SCRIPT}")
if [ -z "${NEXT_JOB_ID}" ]; then
  echo "Failed to queue next dependent job."
  exit 1
fi
echo "Queued next dependent job: ${NEXT_JOB_ID}"

echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V
nvidia-smi || true

"${PYTHON_BIN}" -u "${PROJECT_DIR}/train_addpow.py"
EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
  echo "Training exited with code ${EXIT_CODE}. Cancelling queued next job ${NEXT_JOB_ID}."
  scancel "${NEXT_JOB_ID}" || true
  exit ${EXIT_CODE}
fi

if [ -f "${SOLVED_FILE}" ]; then
  echo "Solved marker found. Cancelling queued next job ${NEXT_JOB_ID}."
  scancel "${NEXT_JOB_ID}" || true
fi
if [ -f "${MAX_FILE}" ]; then
  echo "Hard max marker found. Cancelling queued next job ${NEXT_JOB_ID}."
  scancel "${NEXT_JOB_ID}" || true
fi

exit ${EXIT_CODE}
