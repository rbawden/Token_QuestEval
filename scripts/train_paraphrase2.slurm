#!/bin/bash
#SBATCH -C v100-32g
##SBATCH --partition=gpu_p4
#SBATCH -A ncm@gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G 
#SBATCH --cpus-per-task=8           # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)  
#SBATCH --job-name=safe   # nom du job
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=slurm_outputs/seg%x_%j_%a.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=slurm_outputs/seg%x_%j_%a.out   # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-77%1         # 6 travaux ayant les indices 0, 2, 4, 6, 8, et 10

# go into the submission directory 
cd ${SLURM_SUBMIT_DIR}

maindir=$WORK/Token_QuestEval

module purge
module load cmake/3.14.4
module load cuda/11.2 nccl/2.6.4-1-cuda cudnn/8.1.1.33-cuda
module load intel-mkl/2020.1
module load magma/2.5.4-cuda
module load gcc/10.1.0
module load openmpi/4.1.1-cuda
module load boost/1.74.0

eval "$(/gpfslocalsup/pub/anaconda-py3/2020.02/bin/conda shell.bash hook)"
conda activate py38

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
		    python -u scripts/finetune_t5.py data/paraphrase/parabank2-parts/parabank2.jsonl.part-${SLURM_ARRAY_TASK_ID} --bsz 8 --output_dir models/train_t5_parabank2/ 2>> models/train_t5_parabank2/train.log
