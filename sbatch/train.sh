#!/bin/bash

# load necessary modules or activate your environment
module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa  # If necessary, depends on cluster setup
poetry install  # Install additional Python packages as needed

# Run the application and monitor GPU status in parallel
poetry run python -m src.train