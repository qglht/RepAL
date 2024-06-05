#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=oxfd2547@ox.ac.uk

# load necessary modules or activate your environment
module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa  # If necessary, depends on cluster setup
poetry install  # Install additional Python packages as needed

# Run the application and monitor GPU status in parallel
(poetry run python -m src.pretrain) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 60 seconds until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "Checking GPU status during the application run:"
    nvidia-smi
    sleep 300
done

# Run the application and monitor GPU status in parallel
(poetry run python -m src.train) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 60 seconds until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "Checking GPU status during the application run:"
    nvidia-smi
    sleep 300
done
