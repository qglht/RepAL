#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=RNN_VS_Mamba_dissimilarities_PDM_job
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/miniconda3

source activate repal3


(python -m src.dissimilarity_pairwise_rnnvsmamba --taskset PDM) &

# PID of the application
APP_PID=$!

# Log file path
LOG_FILE=gpu_usage/dissimilarities_rnnvsmamba_PDM_gpu_usage.log

# Monitor GPU status every 600 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    {
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:"
        nvidia-smi
    } >> "$LOG_FILE"  # Append output to log file
    sleep 600 
done

wait $APP_PID