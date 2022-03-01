#!/usr/bin/sh

thisdir=`dirname $0`
maindir=`realpath $thisdir/..`

year=20

[ -d $maindir/scores/wmt$year ] || mkdir $maindir/scores/wmt$year

for type in exact_match_both exact_match_hyp exact_match_ref \
    pred_scores_both pred_scores_hyp pred_scores_ref \
    gold_scores_both gold_scores_hyp gold_scores_ref \
    bert_f_both bert_f_hyp bert_f_ref \
    bert_p_both bert_p_hyp bert_p_ref \
    bert_r_both bert_r_hyp bert_r_ref; do
    metric=${type}-metrics
    [ -d $maindir/scores/wmt$year/$metric ] || mkdir $maindir/scores/wmt$year/$metric
    for langpair in `ls mt-logs/wmt$year/`; do
	echo $langpair
	[ -d $maindir/scores/wmt$year/$metric/$langpair ] || mkdir $maindir/scores/wmt$year/$metric/$langpair
	for scorefile in `ls mt-logs/wmt$year/$langpair/`; do
	    echo -e "\t$scorefile"
	    sysid=`echo $scorefile | perl -pe 's/\.scores\.jsonl//'`

	    if [ -s $maindir/mt-logs/wmt$year/$langpair/$scorefile ]; then
		if [ ! -s $maindir/scores/wmt$year/$metric/$langpair/$sysid.seg ]; then
		    python $thisdir/read_log.py -l seg $maindir/mt-logs/wmt$year/$langpair/$scorefile $type \
			> $maindir/scores/wmt$year/$metric/$langpair/$sysid.seg
		fi
		if [ ! -s $maindir/scores/wmt$year/$metric/$langpair/$sysid.sys ]; then
		    python $thisdir/read_log.py -l sys $maindir/mt-logs/wmt$year/$langpair/$scorefile $type \
			> $maindir/scores/wmt$year/$metric/$langpair/$sysid.sys
		fi
	    fi
	done
    done
done
