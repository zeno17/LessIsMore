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
--output-dir '../models/standard_100M_bert_base' \
--cache-dir '../cached_files' \
--data-dir '../pretraining_data_chunked' \
--book_set 'subset_meta_ratio_100M.pkl'	\
--train_batch_size 32		\
--model-config bert-base \
--training-method 'standard'	\
--sample-size 1	\