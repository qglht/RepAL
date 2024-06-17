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

def generate_and_submit_scripts():
    config = load_config("config.yaml")
    groups = config['groups']  # Assuming 'groups' is a key in your YAML file

    script_template = """
                        #!/bin/bash

                        #SBATCH --nodes=1
                        #SBATCH --time=24:00:00
                        #SBATCH --job-name={group}_job
                        #SBATCH --gres=gpu:8
                        #SBATCH --mail-type=ALL
                        #SBATCH --mail-user=oxfd2547@ox.ac.uk

                        module load cuda/11.2
                        module load pytorch/1.9.0
                        module load python/anaconda3

                        source activate dsa
                        poetry install

                        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory --format=csv,nounits -l 300 > {group}_gpu_usage.log &
                        
                        MONITOR_PID=$!

                        poetry run python src.train_group --group {group}

                        kill $MONITOR_PID
                    """

    for group in groups:
        script_content = script_template.format(group=group)
        script_filename = f"sbatch/{group}_script.sh"

        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)

        # Submit the job to the cluster
        call(f"sbatch {script_filename}", shell=True)

if __name__ == "__main__":
    generate_and_submit_scripts()



