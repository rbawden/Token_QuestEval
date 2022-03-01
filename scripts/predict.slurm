#!/bin/bash
#SBATCH --job-name=bpe-multitok    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:rtx8000:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=20G                # Total memory allocated
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm/gpu_bpe_%A_%a.out   # output file name
#SBATCH --error=slurm/gpu_bpe_%A_%a.out    # error file name
#SBATCH --array=0-20%20

echo "### Running $SLURM_JOB_NAME ###"

#set -x
cd ${SLURM_SUBMIT_DIR}

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    thisscript=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
    thisdir=`dirname $thisscript`
else
    # otherwise: started with bash. Get the real location.
    thisdir=`realpath $(dirname $0)`
fi

module purge
module load gnu8 cuda

maindir=$thisdir/../
year=20
lp=cs-en
dset=test

n=$(($SLURM_ARRAY_TASK_ID  + 1))
num_files=`ls -1 $maindir/data/metrics/wmt$year/hyp/$lp/* | grep -v "-en" | wc -l`

if [ $n -gt $num_files ]; then
    exit
fi

hyp_file=`ls -1 $maindir/data/metrics/wmt$year/hyp/$lp/* | cat -n | sed -n "${n},${n}p" | cut -d'	' -f2`
echo "hyp file = $hyp_file"
echo "slurm num = $n"
langpair=`echo $hyp_file | sed -E "s/^.+?\/hyp\/(.+?)\/news${dset}20${year}.+?$/\1/"`
langpair_nodash=`echo $langpair | sed -E "s/\-//"`
trg=`echo $langpair | cut -d"-" -f2`
sysid=`echo $hyp_file | sed -E "s/^.+?\/${langpair}\/news${dset}20${year}\.(.+?)\.${langpair}/\1/"`
sysid=`echo $hyp_file | perl -pe "s/^.+?\/${langpair}\/news${dset}20${year}.(.+?)\.${langpair}$/\1/" | perl -pe "s/\.$langpair\././"`
ref_file=$maindir/data/metrics/wmt$year/ref/newstest20$year-$langpair_nodash-ref.$trg

echo "langpair = $langpair"
echo "sysid = $sysid"
echo "ref file = $ref_file"
#hyp=$maindir/data/metrics/wmt$year/hyp/newstest2021.HuaweiTSC.ha-en
#ref=$maindir/data/metrics/wmt$year/ref/newstest2021.ha-en.ref.A.en \

[ -d $maindir/mt-logs/wmt$year ] || mkdir $maindir/mt-logs/wmt$year
[ -d $maindir/mt-logs/wmt$year/$langpair ] || mkdir $maindir/mt-logs/wmt$year/$langpair

#echo "TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python $maindir/scripts/predict_t5.py \
#                    $maindir/models/train_t5_metrics/checkpoint-965000/ $hyp_file $ref_file \
#                    > $maindir/mt-logs/wmt$year/$langpair/$sysid.scores.jsonl"
# TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python /scratch/rbawden/Token_QuestEval/scripts/..//scripts/predict_t5.py                     /scratch/rbawden/Token_QuestEval/scripts/..//models/train_t5_metrics/checkpoint-965000/  /scratch/rbawden/Token_QuestEval/scripts/..//data/metrics/wmt20/ref/newstest2020..ref.                     > /scratch/rbawden/Token_QuestEval/scripts/..//mt-logs/wmt20//.scores.jsonl

if [ ! -s $maindir/mt-logs/wmt$year/$langpair/$sysid.scores.jsonl ]; then
    echo "TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u $maindir/scripts/predict_t5.py \
        $maindir/models/train_t5_metrics/checkpoint-965000/ \"$hyp_file\" \"$ref_file\" \
        > $maindir/mt-logs/wmt$year/$langpair/$sysid.scores.jsonl"
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -u $maindir/scripts/predict_t5.py \
	$maindir/models/train_t5_metrics/checkpoint-965000/ "$hyp_file" "$ref_file" \
	> $maindir/mt-logs/wmt$year/$langpair/$sysid.scores.jsonl
fi
