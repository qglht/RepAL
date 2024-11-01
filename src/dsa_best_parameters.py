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


def generate_and_submit_scripts():
    script_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name={n_delay}_{delay_interval}_ordered_job
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/miniconda3

source activate repal3


(python -m src.dsa_optimization --n_delay {n_delay} --delay_interval {delay_interval} --ordered) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 300 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:" >> gpu_usage/{n_delay}_{delay_interval}_ordered_gpu_usage.log
    nvidia-smi >> gpu_usage/{n_delay}_{delay_interval}_ordered_gpu_usage.log  # Append output to log file
    sleep 600 
done

wait $APP_PID


"""
    number_parameters_delays = 10

    n_delays = np.linspace(1, 50, number_parameters_delays, dtype=int)

    for delay in n_delays:
        space = 1
        if not os.path.exists(f"data/dsa_results/20_{delay}_{space}_ordered.csv"):
            script_content = script_template.format(n_delay=delay, delay_interval=space)
            script_filename = f"sbatch/dsa/{delay}_{space}_ordered_script.sh"

            with open(script_filename, "w") as script_file:
                script_file.write(script_content)

            # Submit the job to the cluster
            call(f"sbatch {script_filename}", shell=True)


if __name__ == "__main__":
    generate_and_submit_scripts()
