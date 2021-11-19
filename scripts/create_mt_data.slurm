#!/bin/bash
##SBATCH -C v100-32g
#SBATCH -A ncm@cpu
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2           # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)  
#SBATCH --job-name=safe   # nom du job
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=slurm_outputs/seg%x_%j.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=slurm_outputs/seg%x_%j.out   # Nom du fichier d'erreur (ici commun avec la sortie)
##SBATCH --array=0-232%25         # 6 travaux ayant les indices 0, 2, 4, 6, 8, et 10

# go into the submission directory 
cd ${SLURM_SUBMIT_DIR}

maindir=$WORK/Token_QuestEval
tooldir=/gpfswork/rech/ncm/commun/tools
mosesdir=$tooldir/mosesdecoder
eval "$(/gpfslocalsup/pub/anaconda-py3/2020.02/bin/conda shell.bash hook)"
conda activate py38

bash scripts/create_mt_data_.sh
