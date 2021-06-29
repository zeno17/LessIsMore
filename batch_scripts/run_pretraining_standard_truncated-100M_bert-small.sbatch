#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --get-user-env
#SBATCH --partition=mcs.gpu.q --constraint=v100
#SBATCH --error=../slurm-output/slurm-%j-%x.err
#SBATCH --output=../slurm-output/slurm-%j-%x.out

module load cuda10.2

python3 ../run_pretraining.py \
--output-dir '../models/standard_truncated-100M_bert-small' \
--cache-dir '../cached_files' \
--data-dir '../pretraining_data_truncated' \
--book_set 'subset_meta_ratio_100M.pkl'	\
--train_batch_size 64		\
--model-config bert-small \
--training-method 'standard'	\
--steps-distribution '[1000, 1000, 1000]' \
--sample-size 1	\