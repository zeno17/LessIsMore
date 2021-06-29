#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --get-user-env
#SBATCH --partition=mcs.default.q
#SBATCH --error=../slurm-output/slurm-%j.err
#SBATCH --output=../slurm-output/slurm-%j.out
#SBATCH --time=24:00:00


python3 ../make_optimal_subsets.py --cache-dir '../cached_files'
