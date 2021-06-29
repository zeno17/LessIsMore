#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --get-user-env
#SBATCH --partition=mcs.gpu.q --constraint=v100
#SBATCH --error=../slurm-output/slurm-%j-%x.err
#SBATCH --output=../slurm-output/slurm-%j-%x.out

module load cuda10.2

python3 ../run_glue.py \
--model_name_or_path '../models/standard_100K_bert_tiny/0/' \
--output_dir '../models_finetuned' \
--max_seq_length 128 \
--task_name 'mrpc'