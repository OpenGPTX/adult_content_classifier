#!/bin/bash
#SBATCH --job-name=train-class
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # Total number of tasks
#SBATCH --cpus-per-task=4            # Number of CPUs per task
#SBATCH --mem-per-cpu=10GB           # Mem for CPUs per cpu
#SBATCH --time=10:00:00              # Maximum execution time (HH:MM:SS)
#SBATCH --account=p_gptx
#SBATCH --partition=barnard
#SBATCH --array=1-5  # langs

set -e
source /software/foundation/generic/10_modules.sh

LANGUAGES="en de fr it es"

# Set the working directory
WORKING_DIR=/data/horse/ws/s6690609-gptx_traindata/brandizzi/adult_content_classifier

source "$WORKING_DIR/scripts/barnard/install.sh"
load_modules

language=$(echo $LANGUAGES | cut -d ' ' -f $SLURM_ARRAY_TASK_ID)

## run with slurm array job
poetry -vvv run train --language $language
