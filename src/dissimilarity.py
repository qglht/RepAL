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


def generate_and_submit_scripts(args: argparse.Namespace):
    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name={taskset}_{group1}_VS_{group2}_job
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=small
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

(poetry run python -m src.dissimilarity_pairwise --taskset {taskset} --group1 {group1} --group2 {group2}) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 300 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:" >> gpu_usage/{taskset}_{group1}_VS_{group2}_gpu_usage.log
    nvidia-smi >> gpu_usage/{taskset}_{group1}_VS_{group2}_gpu_usage.log  # Append output to log file
    sleep 300 
done

wait $APP_PID


"""
    config = load_config("config.yaml")
    groups = [
        "pretrain_frozen",
        "pretrain_unfrozen",
        "master",
        "untrained",
        "basic",
    ]

    for i in range(len(groups)):
        group_i = groups[i]
        for j in range(i, len(groups)):
            group_j = groups[j]
            if not os.path.exists(
                f"results/dissimilarity/{args.taskset}/{group_i}_{group_j}.csv"
            ):
                script_content = script_template.format(
                    taskset=args.taskset, group1=group_i, group2=group_j
                )
                script_filename = (
                    f"sbatch/dissimilarity/{args.taskset}/{group_i}_{group_j}_script.sh"
                )

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
