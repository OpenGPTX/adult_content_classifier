#!/bin/bash
#SBATCH --job-name=train-class
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Total number of tasks
#SBATCH --cpus-per-task=8            # Number of CPUs per task
#SBATCH --mem-per-cpu=32GB           # Mem for CPUs per cpu
#SBATCH --time=24:00:00              # Maximum execution time (HH:MM:SS)
#SBATCH --account=p_gptx
#SBATCH --partition=alpha
#SBATCH --output=/data/horse/ws/anbh299g-logs/logs/adult_content_classifier/slurm_logs/%A_%a.out.log
#SBATCH --error=/data/horse/ws/anbh299g-logs/logs/adult_content_classifier/slurm_logs/%A_%a.err.log
##SBATCH --gres=gpu:4




#module purge

set -e
#source /software/foundation/generic/10_modules.sh

# Set the working directory
WORKING_DIR=/home/anbh299g/code/adult_content_classifier

# source "$WORKING_DIR/scripts/barnard/install.sh"
# load_modules

source /software/foundation/generic/10_modules.sh
# Load the required modules
# Update the version of the modules if needed
module --force purge
module load development/24.04  GCCcore/13.2.0
module load Python/3.11.5

source $WORKING_DIR/venv_roberta/bin/activate
echo "Activated virtual environment"
#language="en"
# Activate Poetry virtual environment (adjust path accordingly)
#source $(poetry env info --path)/bin/activate

## run with slurm array job
#poetry run train_roberta --input_dir /data/horse/ws/s6690609-gptx_traindata/raw_data/cc/cc_wet_dumps_converted_dt/ --output_path /data/horse/ws/s6690609-gptx_traindata/anirban/adult_content_classifier/artifacts --model_name xlm-roberta-large
python $WORKING_DIR/adult_content_classifier/cli.py --input_dir /data/horse/ws/s6690609-gptx_traindata/raw_data/cc/cc_wet_dumps_converted_dt/ --language en --language de --language fr --language it --language es --output_path /data/horse/ws/s6690609-gptx_traindata/anirban/adult_content_classifier/artifacts --model_name xlm-roberta-large
# srun python $WORKING_DIR/adult_content_classifier/cli.py