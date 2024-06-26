#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dissimilarity_over_learning
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

(poetry run python -m src.dissimilarity_over_learning --group1 pretrain_frozen --group2 pretrain_unfrozen) & 

# PID of the application
APP_PID=$!

# Monitor GPU status every 300 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:" >> gpu_usage/learning_${group1}_${group2}_gpu_usage.log
    nvidia-smi >> gpu_usage/learning_${group1}_${group2}_gpu_usage.log  # Append output to log file
    sleep 300 
done

wait $APP_PID