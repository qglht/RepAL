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
from src.train_group import train


def generate_and_submit_scripts(args: argparse.Namespace):
    config = load_config("config.yaml")
    groups = config[args.taskset][
        "groups"
    ]  # Assuming 'groups' is a key in your YAML file

    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name={taskset}_{group}_job
#SBATCH --output=slurm-{taskset}_{group}.out    # Output file
#SBATCH --error=slurm-{taskset}_{group}.err     # Error file
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

(poetry run python -m src.train_group --taskset {taskset} --group {group}) & 

# PID of the application
APP_PID=$!

# Log file path
LOG_FILE=gpu_usage/{taskset}_{group}_gpu_usage.log

# Monitor GPU status every 600 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    {{
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:"
        nvidia-smi
    }} >> "$LOG_FILE"  # Append output to log file
    sleep 600 
done

wait $APP_PID
"""
    for group in groups:
        script_content = script_template.format(taskset=args.taskset, group=group)
        script_filename = f"sbatch/groups/{args.taskset}_{group}_script.sh"

        with open(script_filename, "w") as script_file:
            script_file.write(script_content)

        # Submit the job to the cluster
        call(f"sbatch {script_filename}", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--taskset",
        type=str,
        default="PDM",
        help="The taskset to train the model on",
    )
    args = parser.parse_args()
    generate_and_submit_scripts(args)
