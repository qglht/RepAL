#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=master_job
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80  # 10 CPUs per GPU * 8 GPUs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory --format=csv,nounits -l 300 > gpu_usage/master_gpu_usage.log &
nvidia-smi pmon -c 1 -s um > gpu_usage/master_gpu_processes.log &

MONITOR_PID=$!

poetry run python -m src.train_group --group master

kill $MONITOR_PID
