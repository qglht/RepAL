import numpy as np
import os
import sys
from subprocess import call
import warnings
import argparse
from dsa_analysis import load_config
import torch
import multiprocessing
from src.toolkit import pipeline
from src.dsa_optimization import dsa_computation

def generate_and_submit_scripts():
    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --job-name={group1}_VS_{group2}_job
#SBATCH --gres=gpu:1
#SBATCH --partition=small
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

(poetry run python -m src.dissimilarity_pairwise --group1 {group1} --group2 {group2}) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 300 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:" >> gpu_usage/{group1}_VS_{group2}_gpu_usage.log
    nvidia-smi >> gpu_usage/{group1}_VS_{group2}_gpu_usage.log  # Append output to log file
    sleep 300 
done

wait $APP_PID


"""
    config = load_config("config.yaml")
    groups = ["anti", "basic","delay","master","pretrain_frozen","pretrain_unfrozen","untrained"]

    for i in range(len(groups)):
        group_i = groups[i]
        for j in range(i, len(groups)):
            group_j = groups[j]
            script_content = script_template.format(group1=group_i, group2=group_j)
            script_filename = f"sbatch/dissimilarity/{group_i}_{group_j}_script.sh"

            with open(script_filename, 'w') as script_file:
                script_file.write(script_content)

            # Submit the job to the cluster
            call(f"sbatch {script_filename}", shell=True)

if __name__ == "__main__":
    generate_and_submit_scripts()