import os
from subprocess import call
import argparse


def generate_and_submit_scripts(args: argparse.Namespace):
    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=dissimilarity_over_learning_{taskset}_{group1}_{group2}
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/miniconda3

source activate repal3


(python -m src.dissimilarity_over_learning_per_group --taskset {taskset} --group1 {group1} --group2 {group2}) & 

# PID of the application
APP_PID=$!

# Log file path
LOG_FILE=gpu_usage/learning_{taskset}_{group1}__{group2}_gpu_usage.log

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
    groups = [
        "untrained",
        "master_frozen",
        "master",
        "pretrain_basic_frozen",
        "pretrain_anti_frozen",
        "pretrain_delay_frozen",
        "pretrain_basic_anti_frozen",
        "pretrain_basic_delay_frozen",
        "pretrain_frozen",
        "pretrain_unfrozen",
    ]

    for i in range(len(groups) - 1, -1, -1):
        if groups[i] == "master":
            for j in range(len(groups) - 1, -1, -1):
                group = [groups[i], groups[j]]
                script_content = script_template.format(
                    taskset=args.taskset, group1=group[0], group2=group[1]
                )
                script_filename = f"sbatch/dissimilarities_over_learning/{args.taskset}/{group[0]}_{group[1]}_script.sh"

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
        help="taskset to compare dissimilarities on",
    )
    args = parser.parse_args()
    generate_and_submit_scripts(args)
