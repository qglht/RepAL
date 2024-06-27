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
#SBATCH --job-name={n_delay}_{delay_interval}_{ordered}_job
#SBATCH --gres=gpu:1
#SBATCH --partition=small
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

(poetry run python -m src.dsa_optimization --n_delay {n_delay} --delay_interval {delay_interval} --ordered {ordered} --overwrite True) &

# PID of the application
APP_PID=$!

# Monitor GPU status every 300 seconds (5 minutes) until the application finishes
while kill -0 $APP_PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking GPU status during the application run:" >> gpu_usage/{n_delay}_{delay_interval}_{ordered}_gpu_usage.log
    nvidia-smi >> gpu_usage/{n_delay}_{delay_interval}_{ordered}_gpu_usage.log  # Append output to log file
    sleep 300 
done

wait $APP_PID


"""
    number_parameters_delays = 10
    number_parameters_intervals = 10

    n_delays = np.linspace(1, 50, number_parameters_delays, dtype=int)
    delay_interval = np.linspace(1, 50, number_parameters_intervals, dtype=int)

    for delay in n_delays:
        for space in delay_interval:
            if space < int(200/delay):
                for ordered in [True, False]:
                    path_file = f'data/dsa_results/50_{delay}_{space}.csv' if not ordered else f'data/dsa_results/50_{delay}_{space}_ordered.csv'
                    if not os.path.exists(path_file):
                        script_content = script_template.format(n_delay=delay, delay_interval=space, ordered=ordered)
                        script_filename = f"sbatch/dsa/{delay}_{space}_{ordered}_script.sh"

                        with open(script_filename, 'w') as script_file:
                            script_file.write(script_content)

                        # Submit the job to the cluster
                        call(f"sbatch {script_filename}", shell=True)

if __name__ == "__main__":
    generate_and_submit_scripts()