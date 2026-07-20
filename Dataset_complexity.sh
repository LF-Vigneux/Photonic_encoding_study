#!/bin/bash
#SBATCH --job-name=Dataset_complexity
#SBATCH --output=Dataset_complexity_%j.out
#SBATCH --error=Dataset_complexity_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mem=100000

module load python/3.12.4

source $HOME/venv/Encoding_Study_official/bin/activate

cd $SLURM_SUBMIT_DIR

python -u implementation.py --config "configs/induced_dataset_complexity_exp.json"