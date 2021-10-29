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

eval "$(/gpfslocalsup/pub/anaconda-py3/2020.02/bin/conda shell.bash hook)"
conda activate py38

# get initial examples
python  data/paraphrase/get_one_per_example.py data/paraphrase/full/parabank.tsv \
	data/paraphrase/full/parabank.meta | gzip > data/paraphrase/parabank.1perexample.threshold0.7.tsv.gz 

# tokenise
paste <(zcat data/paraphrase/parabank.1perexample.threshold0.7.tsv.gz | cut -f 1 | perl scripts/dag2txt -l en -nffs -nfc -no_a ) \
      <(zcat data/paraphrase/parabank.1perexample.threshold0.7.tsv.gz | cut -f 2 | perl scripts/dag2txt -l en -nffs -nfc -no_a ) \
    | gzip > data/paraphrase/parabank.1perexample.threshold0.7.detok.tsv.gz

# mask examples
zcat data/paraphrase/parabank.1perexample.threshold0.7.detok.tsv.gz \
    | python -u scripts/create_masked_examples.py \
	     > data/paraphrase/parabank.threshold0.7.detok.masked-examples.jsonl
