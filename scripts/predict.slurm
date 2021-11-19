#!/bin/bash
##SBATCH -C v100-32g
#SBATCH -A ncm@gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=36G  
#SBATCH --cpus-per-task=2           # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)  
#SBATCH --job-name=safe   # nom du job
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=seg%x_%j.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=seg%x_%j.out   # Nom du fichier d'erreur (ici commun avec la sortie)
##SBATCH --array=0-232%25         # 6 travaux ayant les indices 0, 2, 4, 6, 8, et 10

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

#bash scripts/metric-score-ref.sh mask-base-hyp-ref
echo -e "Oh , darling , what happened ? \t Oh , honey , what 's the matter with you ?" | TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u scripts/predict.py > out

#paste <(head data/metrics/wmt20/hyp/de-en/newstest2020.Huoshan_Translate.789.de-en.en) <(head data/metrics/wmt20/ref/newstest2020-deen-ref.en) | \
#    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u scripts/predict.py > out
