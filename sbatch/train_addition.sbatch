#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=train_addition
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
source /home/padia.so/Workspace/grokking-operators/.venv/bin/activate

cd /home/padia.so/Workspace/grokking-operators
mkdir -p logs
python train_addition.py
