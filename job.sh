#!/bin/bash
#SBATCH --job-name=train_noun_chunker
#SBATCH --output=/home/mila/y/yu-lu.liu/noun_chunker/job_output.txt
#SBATCH --error=/home/mila/y/yu-lu.liu/noun_chunker/job_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem=32G
#SBATCH --partition=unkillable
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:32gb:1

module load python/3.5

source env/bin/activate

python create_token_cnn_dataset.py