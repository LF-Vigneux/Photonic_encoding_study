#!/bin/bash
#SBATCH --job-name=Dataset_complexity
#SBATCH --output==Dataset_complexity_%j.out
#SBATCH --error==Dataset_complexity_%j.err
#SBATCH --time=13:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=60000

module load python/3.11.5
module load symengine/0.13.0

source $HOME/venv/TEMS_qiskit_1/bin/activate

cd $SLURM_SUBMIT_DIR

python -u runner.py --config "configs/induced_dataset_complexity_exp.json"