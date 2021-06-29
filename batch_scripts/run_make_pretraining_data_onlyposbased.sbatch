#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --get-user-env
#SBATCH --partition=mcs.default.q
#SBATCH --error=../slurm-output/slurm-%j-%x.err
#SBATCH --output=../slurm-output/slurm-%j-%x.out

python3 ../make_pretraining_data.py --cache-dir '../cached_files' --data-dir '../pretraining_data_onlyposbased' --book_id_file 'subset_meta_ratio_100M.pkl' --truncation 'chunk' --split-sizes [8,32,128] --lemmatize False --ner-based-swap False