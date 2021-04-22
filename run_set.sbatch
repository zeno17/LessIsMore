#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --get-user-env
#SBATCH --partition=mcs.gpu.q
#SBATCH --error=slurm-%jerr
#SBATCH --output=slurm-%j.out
#SBATCH --time=2:00:00

#module load anaconda3
#module load cuda10.1/toolkit/10.1.243
#module load cuda10.1

python3
./run_demo.py
