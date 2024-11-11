#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=compress_data_models_job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxfd2547@ox.ac.uk

# Load necessary modules, if required
module load zip

# Compress the data folder
zip -r data.zip data/

# Compress the models folder
zip -r models.zip models/
