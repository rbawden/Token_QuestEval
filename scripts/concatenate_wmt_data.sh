#!/bin/sh
thisdir=`realpath $(dirname $0)`
maindir=$thisdir/..
data_folder=$maindir/data/metrics


for year in 14 15 16 17 18; do
    # go through all into-English reference files
    for ref in `ls $data_folder/wmt$year/ref/newstest20$year*.en`; do
	# get language pair (without and with the dash)
	lp=`echo $ref | perl -pe 's/^.+?(....)-ref.en$/\1/'` # e.g. fien
	lp_dash=`echo $lp | perl -pe 's/(..)(en)/\1-\2/'` # e.g. fi-en
	# for each mt submission for this language pair, print out reference and submission
	for mt_sub in `ls $data_folder/wmt$year/hyp/$lp_dash/newstest20$year*en`; do
	    paste $mt_sub $ref 
	done
    done
done
