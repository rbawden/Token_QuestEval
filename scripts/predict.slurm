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
##SBATCH --array=1-340%25

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

year=20
dset=test

n=1 #$SLURM_ARRAY_TASK_ID 

hyp_file=`ls -1 $maindir/data/metrics/wmt$year/hyp/*-en/* | cat -n | sed -n "${n},${n}p" | cut -d'	' -f2`
echo $hyp_file
langpair=`echo $hyp_file | sed -E "s/^.+?\/hyp\/(.+?)\/news$dset20$year.+?$/\1/"`
trg=`echo $langpair | cut -d"-" -f2`
sysid=`echo $hyp_file | sed -E "s/^.+?\/$lp\/news$dset20$year\.(.+?)\.$lp/\1/"`
ref_file=$maindir/data/metrics/wmt$year/ref/newstest20$year.$lp.ref.$trg


echo $ref_file
#hyp=$maindir/data/metrics/wmt$year/hyp/newstest2021.HuaweiTSC.ha-en
#ref=$maindir/data/metrics/wmt$year/ref/newstest2021.ha-en.ref.A.en \

[ -d $maindir/scores/metric/wmt$year ] || mkdir -p $maindir/scores/metric/wmt$year

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python $maindir/scripts/predict_t5.py \
		    $maindir/models/train_t5_parabank1/checkpoint-395000/ "$hyp_file" "$ref_file" \
		    > $maindir/scores/metric/wmt$year/$dset.$sysid.$langpair.all-scores

