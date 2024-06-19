#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name={group}_job
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10  # 10 CPUs per GPU * 8 GPUs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

# load necessary modules or activate your environment
module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa  # If necessary, depends on cluster setup
poetry install  # Install additional Python packages as needed

# Check GPU status before running the application
echo "Checking GPU status before running the application:"
nvidia-smi

# Run the application and monitor GPU status in parallel
(poetry run python -m src.test --group pretrain_unfrozen) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 60 seconds until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "Checking GPU status during the application run:"
    nvidia-smi
    sleep 60
done
