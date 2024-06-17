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
#SBATCH --time=4:00:00
#SBATCH --job-name={n_delay}_{delay_interval}_job
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa
poetry install

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory --format=csv,nounits -l 300 > gpu_usage/dsa/{group}_gpu_usage.log &

MONITOR_PID=$!

poetry run python -m src.dsa_optimization --n_delay {n_delay} --delay_interval {delay_interval}

kill $MONITOR_PID
"""
    number_parameters_delays = 10
    number_parameters_intervals = 5

    n_delays = np.linspace(5, 50, number_parameters_delays, dtype=int)

    for delay in n_delays:
        delay_interval = np.linspace(1, int(200/delay), number_parameters_intervals, dtype=int)
        for space in delay_interval:
            script_content = script_template.format(n_delay=delay, delay_interval=space)
            script_filename = f"sbatch/dsa/{delay}_{space}_script.sh"

            with open(script_filename, 'w') as script_file:
                script_file.write(script_content)

            # Submit the job to the cluster
            call(f"sbatch {script_filename}", shell=True)

if __name__ == "__main__":
    generate_and_submit_scripts()