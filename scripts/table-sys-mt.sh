#!/usr/bin/env bash

# Used to score metric outputs that have already been produced, and
# make a nice table.  You should be in a "scores" directory, with a
# directory structure as described in the "Scoring" section in the
# README.
#
# Run:
#
#    bash table.sh {wmt}
#
# where {wmt} is e.g., "wmt19". You may have to first fix paths.
# The script assumes that the repo "data" directory is a sister of the
# current directory.
thisdir=`dirname $0`
wmt=${1:-wmt19}

wmt_list=$(sacrebleu -t $wmt --list | tr " " "\n" | sort | tr "\n" " " ) # avoids change in order in certain environemnts
wmt_list=$(sacrebleu -t $wmt --list | tr " " "\n" | grep -P ".*?-en" | sort | tr "\n" " " ) # avoids change in order in certain environemnts
echo -e -n "$wmt\t\t\t"; for pair in $wmt_list; do echo -ne "\t$pair"; done; echo;
for metric in $(cd $thisdir/../scores/$wmt; ls); do
    displayname=$(echo "$metric                                         " | cut -c1-25)
    #displayname=$(echo "$metric vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" | cut -c1-25)
    echo -n "$displayname"

    for pair in $wmt_list; do #$(sacrebleu -t $wmt --list | sort); do
        # Create a cache of the scores for this metric, so future runs are fast
        cachefile=$thisdir/../scores/$wmt/$metric/$pair/cache.sys
        if [[ -s $cachefile ]]; then
            # use the cache if it exists
            score=$(cat $cachefile)
        else
            # if the cache isn't there, create it
	    if [ -e $thisdir/../data/metrics/$wmt/gold/sys/$pair.csv ]; then
		#echo "python3 $thisdir/corr-folder.py $thisdir/../data/metrics/$wmt/gold/sys/$pair.csv \
                #    $thisdir/../scores/$wmt/$metric/$pair"
		#read
                score=$(python3 $thisdir/corr-folder.py $thisdir/../data/metrics/$wmt/gold/sys/$pair.csv \
		    $thisdir/../scores/$wmt/$metric/$pair)
            else
                score="-"
            fi
	    [ -d $thisdir/../scores/$wmt/$metric/$pair ] || mkdir $thisdir/../scores/$wmt/$metric/$pair
            [[ $score != "ERR" ]] && echo $score > $cachefile
        fi
        echo -ne "\t$score"
    done
    echo
done
