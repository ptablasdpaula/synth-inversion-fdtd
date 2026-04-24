#!/bin/bash
#SBATCH --job-name=ness-flow-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── Environment ──
eval "$(pixi shell-hook)"
# or: source activate ness-flow

mkdir -p logs

echo "Starting NESS flow matching training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Param2Tok equivariant (main experiment) ──
python train.py experiment=ness_param2tok \
    data.data_dir=./data/ness_guitar \
    data.num_workers=8 \
    trainer.devices=1

echo "Training complete: $(date)"