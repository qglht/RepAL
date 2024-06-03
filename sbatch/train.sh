#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1:00:00

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=oxfd2547@ox.ac.uk

# load necessary modules or activate your environment
module load cuda/11.2
module load pytorch/1.9.0
module load python/anaconda3

source activate dsa  # If necessary, depends on cluster setup
poetry install  # Install additional Python packages as needed
poetry update

# Define a function to monitor GPU usage
monitor_gpu_usage() {
    python - <<EOF
import time
import subprocess
import matplotlib.pyplot as plt
from threading import Thread

def get_gpu_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    return int(result.stdout.decode('utf-8').strip().split('\n')[0])

def plot_gpu_usage(gpu_usage_list):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU Usage (%)')
    line, = ax.plot(gpu_usage_list, 'r-')

    while True:
        line.set_ydata(gpu_usage_list)
        line.set_xdata(range(len(gpu_usage_list)))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.3)

gpu_usage_list = []

# Start plotting thread
plot_thread = Thread(target=plot_gpu_usage, args=(gpu_usage_list,))
plot_thread.start()

try:
    while True:
        gpu_usage = get_gpu_usage()
        gpu_usage_list.append(gpu_usage)
        time.sleep(0.3)
except KeyboardInterrupt:
    pass

plot_thread.join()
EOF
}

# Start GPU monitoring in the background
monitor_gpu_usage &

# Run the pretrain script and wait for it to complete
poetry run python -m src.pretrain

# Run the train script after pretrain completes
poetry run python -m src.train

# Wait for all background processes to complete
wait
