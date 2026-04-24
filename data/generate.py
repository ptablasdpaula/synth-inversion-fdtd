#!/bin/bash
#SBATCH --job-name=ness-datagen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/datagen_%j.out
#SBATCH --error=logs/datagen_%j.err

# ── Environment ──
# Adjust to your pixi/conda env
eval "$(pixi shell-hook)"
# or: source activate ness-flow

mkdir -p logs

echo "Starting NESS dataset generation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python generate_dataset.py \
    --output_dir ./data/ness_guitar \
    --num_samples 200000 \
    --ness_binary ./ness/ness-framework \
    --num_workers ${SLURM_CPUS_PER_TASK} \
    --seed 42

echo "Dataset generation complete"
echo "Computing mel statistics..."

python compute_stats.py \
    --data_dir ./data/ness_guitar \
    --num_samples 10000

echo "Done: $(date)"