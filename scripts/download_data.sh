#!/bin/sh

thisdir=`realpath $(dirname $0)`
maindir=$thisdir/..
echo "Main project directory = $maindir"

[ -d $maindir/data ] || mkdir $maindir/data
[ -d $maindir/data/paraphrase ] || mkdir $maindir/data/paraphrase
[ -d $maindir/data/metrics ] || mkdir $maindir/data/metrics

# Parabank v1
if [ ! -f $maindir/data/paraphrase/parabank1.tsv ]; then
    # download full parabank
    wget http://cs.jhu.edu/~vandurme/data/parabank-1.0-full.zip -O $maindir/data/paraphrase/parabank-1.0-full.zip
    # unzip it
    unzip $maindir/data/paraphrase/parabank-1.0-full.zip -d $maindir/data/paraphrase/
    # delete zipped data
    rm $maindir/data/paraphrase/parabank-1.0-full.zip
    # move data out of folder
    mv $maindir/data/paraphrase/full/parabank.tsv $maindir/data/paraphrase/parabank1.tsv
    mv $maindir/data/paraphrase/full/parabank.meta $maindir/data/paraphrase/parabank1.meta
fi

# Parabank v2
if [ ! -f $maindir/data/paraphrase/parabank2.tsv ]; then
    # download full parabank
    wget http://cs.jhu.edu/~vandurme/data/parabank-2.0 -O $maindir/data/paraphrase/parabank-2.0.zip
    # unzip it
    unzip $maindir/data/paraphrase/parabank-2.0.zip -d $maindir/data/paraphrase/
    # delete zipped data
    rm $maindir/data/paraphrase/parabank-2.0.zip
fi

# WMT submission data
if [ ! -f $maindir/data/metrics/wmt14-submitted-data.tgz ]; then
    wget http://www.statmt.org/wmt14/submissions.tgz -O $maindir/data/metrics/wmt14-submitted-data.tgz
fi
if [ ! -f $maindir/data/metrics/wmt15-submitted-data.tgz ]; then
    wget http://www.statmt.org/wmt15/wmt15-submitted-data.tgz -O $maindir/data/metrics/wmt15-submitted-data.tgz
fi
if [ ! -f $maindir/data/metrics/wmt16-submitted-data.tgz ]; then
    wget http://data.statmt.org/wmt16/translation-task/wmt16-submitted-data-v2.tgz -O $maindir/data/metrics/wmt16-submitted-data.tgz
fi
if [ ! -f $maindir/data/metrics/wmt17-submitted-data.tgz ]; then
    wget http://data.statmt.org/wmt17/translation-task/wmt17-submitted-data-v1.0.tgz -O $maindir/data/metrics/wmt17-submitted-data.tgz
fi
if [ ! -f $maindir/data/metrics/wmt18-submitted-data.tgz ]; then
    wget http://data.statmt.org/wmt18/translation-task/wmt18-submitted-data-v1.0.1.tgz -O $maindir/data/metrics/wmt18-submitted-data.tgz
fi
if [ ! -f $maindir/data/metrics/wmt19-submitted-data.tgz ]; then
    wget http://data.statmt.org/wmt19/translation-task/wmt19-submitted-data-v3.tgz -O $maindir/data/metrics/wmt19-submitted-data.tgz
fi
if [ ! -f $maindir/data/metrics/wmt20-submitted-data.tgz ]; then
    gdown https://drive.google.com/uc?id=1Wjn8AaQae6dcvd8oK7Qu4m8jITJ4g5o5 -O $maindir/data/metrics/wmt20-submitted-data.tgz
fi

if [ ! -f $maindir/data/metrics/wmt21-submitted-data.tgz ]; then
    wget https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.0.5.tar.gz -O $maindir/data/metrics/wmt21-submitted-data.tgz
fi

for year in 14 15 16 17 18 19 20 21; do
    if [ ! -f $maindir/data/metrics/.wmt$year-submitted-data.tgz ]; then
	tar -xzvf $maindir/data/metrics/wmt$year-submitted-data.tgz -C $maindir/data/metrics/
	[ -d $maindir/data/metrics/wmt$year ] || mkdir $maindir/data/metrics/wmt$year
	for folder in hyp ref src; do
	    [ -d $maindir/data/metrics/wmt$year/$folder ] || mkdir $maindir/data/metrics/wmt$year/$folder
	done
	
	if [ "$year" -eq "14" ] && [ ! -d $maindir/data/metrics/wmt14-submitted-data ]; then
	    mv $maindir/data/metrics/wmt14-data $maindir/data/metrics/wmt14-submitted-data
	fi
	if [ "$year" -eq "20" ] && [ ! -f $maindir/data/metrics/wmt20-submitted-data ]; then
	    mv $maindir/data/metrics/wmt20metricsdata-v2 $maindir/data/metrics/wmt20-submitted-data
	    mv $maindir/data/metrics/wmt20-submitted-data/newstest2020/txt/sources/newstest20* $maindir/data/metrics/wmt20/src/
	    mv $maindir/data/metrics/wmt20-submitted-data/newstest2020/txt/references/newstest20* $maindir/data/metrics/wmt20/ref/
	    mv $maindir/data/metrics/wmt20-submitted-data/newstest2020/txt/system-outputs/* $maindir/data/metrics/wmt20/hyp/
	    for file in $maindir/data/metrics/wmt20/ref/*; do
		mv $file ${file%.txt}
	    done
	    for file in $maindir/data/metrics/wmt20/src/*; do
		mv $file ${file%.txt}
	    done
	    for lp in $maindir/data/metrics/wmt20/hyp/*; do
		for file in $lp/newstest2020.*; do
		    echo $file;
		done
	    done
	    for lp in `ls $maindir/data/metrics/wmt20/hyp/`; do
		for file in `ls $maindir/data/metrics/wmt20/hyp/$lp/newstest2020.*`; do
		    new=`echo ${file%.txt}.$lp | perl -pe "s/newstest2020\.$lp\./newstest2020./"`
		    mv $file $new
		done
	    done
	elif [ "$year" -eq "21" ] && [ ! -f $maindir/data/metrics/wmt21-submitted-data ]; then
	    mv $maindir/data/metrics/wmt21-news-systems-1.0.5 $maindir/data/metrics/wmt21-submitted-data
	    mv $maindir/data/metrics/wmt21-submitted-data/txt/references/* $maindir/data/metrics/wmt21/ref/
	    mv $maindir/data/metrics/wmt21-submitted-data/txt/sources/* $maindir/data/metrics/wmt21/src/
	    mv $maindir/data/metrics/wmt21-submitted-data/txt/system-outputs/* $maindir/data/metrics/wmt21/hyp/
	    for file in $maindir/data/metrics/wmt21/hyp/*test*; do
		lp=`echo $file | perl -pe 's/^.+?test2021\.(.+?)\..+?$/\1/'`
		trg=`echo $lp | cut -d"-" -f2`
		[ -d $maindir/data/metrics/wmt21/hyp/$lp ] || mkdir $maindir/data/metrics/wmt21/hyp/$lp
		new=`echo $file | perl -pe "s/newstest2021\.$lp\.hyp\.(.+?).$trg/newstest2021.\1.$lp/"`
		mv $file $new
	    done
	else
	    mv $maindir/data/metrics/wmt$year-submitted-data/txt/references/newstest20$year-* $maindir/data/metrics/wmt$year/ref/
	    mv $maindir/data/metrics/wmt$year-submitted-data/txt/sources/newstest20$year-* $maindir/data/metrics/wmt$year/src/
	    mv $maindir/data/metrics/wmt$year-submitted-data/txt/system-outputs/newstest20$year/* $maindir/data/metrics/wmt$year/hyp/
	fi
	touch $maindir/data/metrics/.wmt$year-submitted-data.tgz
    fi
done

