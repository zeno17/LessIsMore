#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --get-user-env
#SBATCH --partition=mcs.gpu.q --constraint=v100
#SBATCH --exclusive
#SBATCH --error=../slurm-output/slurm-%j-%x.err
#SBATCH --output=../slurm-output/slurm-%j-%x.out

module load cuda10.2

python3 ../run_pretraining.py \
--output-dir '../models/truncated_standard_100M' \
--cache-dir '../cached_files' \
--data-dir '../pretraining_data_truncated' \
--book_set 'subset_meta_ratio_100M.pkl'	\
--train_batch_size 32	\
--model-config bert-base \
--training-method 'standard'	\
--steps-distribution '[10000, 40000, 50000]' \
--sample-size 1

python3 ../run_pretraining.py \
--output-dir '../models/singlelength_standard_100M' \
--cache-dir '../cached_files' \
--data-dir '../pretraining_data_singlelength' \
--book_set 'subset_meta_ratio_100M.pkl'	\
--train_batch_size 32	\
--model-config bert-base \
--training-method 'single_length' \
--steps-distribution '[10000, 40000, 50000]' \
--sample-size 1

python3 ../run_pretraining.py \
--output-dir '../models/posbased_standard_100M' \
--cache-dir '../cached_files' \
--data-dir '../pretraining_data_onlyposbased' \
--book_set 'subset_meta_ratio_100M.pkl'	\
--train_batch_size 32	\
--model-config bert-base \
--training-method 'standard'	\
--steps-distribution '[10000, 40000, 50000]' \
--sample-size 1

python3 ../run_pretraining.py \
--output-dir '../models/lemmatized_standard_100M' \
--cache-dir '../cached_files' \
--data-dir '../pretraining_data_onlylemmatized' \
--book_set 'subset_meta_ratio_100M.pkl'	\
--train_batch_size 32	\
--model-config bert-base \
--training-method 'standard'	\
--steps-distribution '[10000, 40000, 50000]' \
--sample-size 1
