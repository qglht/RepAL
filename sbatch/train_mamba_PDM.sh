#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=1:00:00
# set name of job
#SBATCH --job-name=MASTERJOB
# set number of GPUs
#SBATCH --gres=gpu:0
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=qguilhot@gmail.com

# load necessary modules or activate your environment
module load cuda/11.2
module load pytorch/1.9.0
module load python/miniconda3

source activate repal3  # If necessary, depends on cluster setup
  # Install additional Python packages as needed

# Run the application and monitor GPU status in parallel
python -m src.train_mamba --taskset PDM
