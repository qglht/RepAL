import os
from subprocess import call
import argparse


def generate_and_submit_scripts(args: argparse.Namespace):
    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=dissimilarity_within_learning_{taskset}_{group}
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/miniconda3

source activate repal3


(python -m src.dissimilarity_within_learning_per_group --taskset {taskset} --group {group}) & 

# PID of the application
APP_PID=$!

# Log file path
LOG_FILE=gpu_usage/dissimilarity_within_learning_{taskset}_{group}_gpu_usage.log

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
    groups_within_learning = [
        "pretrain_frozen",
        "pretrain_unfrozen",
        "master",
        "untrained",
    ]

    for group in groups_within_learning:
        script_content = script_template.format(taskset=args.taskset, group=group)
        script_filename = (
            f"sbatch/dissimilarities_within_learning/{args.taskset}/{group}_script.sh"
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
        help="taskset to compare dissimilarities on",
    )
    args = parser.parse_args()
    generate_and_submit_scripts(args)
