#!/bin/bash
#SBATCH --job-name=tabfight
#SBATCH --output=results/logs/%j.out
#SBATCH --error=results/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu

# activate your conda environment
source activate tabfight

# go to project directory
cd ~/TabFight

# run benchmark
python scripts/run_benchmark.py