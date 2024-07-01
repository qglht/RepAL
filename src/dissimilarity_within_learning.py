import os
from subprocess import call


def generate_and_submit_scripts():
    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dissimilarity_within_learning_{group}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

(poetry run python -m src.dissimilarity_within_learning_per_group --group {group}) & 

# PID of the application
APP_PID=$!

# Monitor GPU status every 300 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:" >> gpu_usage/learning_within_{group}_gpu_usage.log
    nvidia-smi >> gpu_usage/learning_within_{group}_gpu_usage.log  # Append output to log file
    sleep 300 
done

wait $APP_PID

"""
    groups_within_learning = ["pretrain_frozen", "pretrain_unfrozen", "master"]

    for group in groups_within_learning:
        script_content = script_template.format(group=group)
        script_filename = f"sbatch/dissimilarities_within_learning/{group}_script.sh"

        with open(script_filename, "w") as script_file:
            script_file.write(script_content)

        # Submit the job to the cluster
        call(f"sbatch {script_filename}", shell=True)


if __name__ == "__main__":
    generate_and_submit_scripts()