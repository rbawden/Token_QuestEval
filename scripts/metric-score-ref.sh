#!/bin/bash

thisdir=`realpath $(dirname $0)`
maindir=$thisdir/..

metric_name=$1

# check parameters
if [[ "$#" -ne 1 ]]; then
    echo "Error: expected 1 param"
    echo "Usage: $0 <metric_name>"
    exit
fi

year=19
testset=newstest20$year

# go through all language pairs in the submissions folder for the specific testset
#`ls -d $maindir/data/hyp/$testset/* | rev | cut -d/ -f1 | rev`; do
for lp in  de-en; do 
    slang=${lp:0:2}; tlang=${lp:3:5}
    
    # get the source file
    src=$maindir/data/metrics/wmt$year/src/$testset-$slang$lang-src.$tlang

    echo $src
    # for each of the reference files for this language pair
    for ref in $maindir/data/metrics/wmt$year/ref/$testset-$slang$tlang-ref.$tlang; do
        basename=$(basename $ref);
	refset=$(echo $basename | cut -d- -f1);

	# for each of the MT hypotheses (i.e. the submissions to the news task)
	# each of the files must finished with $tlang, otherwise it won't be considered (e.g. for postprocessed files)
        for hyp in $maindir/data/metrics/wmt$year/hyp/$lp/*$tlang; do
	    basename=$(basename $hyp);
	    systemname=$(echo $basename | perl -pe "s/$testset\.(.+?)\.$lp\.$tlang/\1/")

	    # ignore certain systems for newstest2020
            if [ $refset = newstest2020 ] && [ $systemname = 'Human-A.0' ]; then continue; fi
            if [ $refset = newstestB2020 ] && [ $systemname = 'Human-B.0' ]; then continue; fi
            if [ $refset = newstestP2020 ] && [ $systemname = 'Human-P.0' ]; then continue; fi
	    
	    # call the metric scores
	    paste $hyp $ref | TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/predict.py > scores/wmt19/logs/${metric_name}.$systemname.log
        done  
    done
done

