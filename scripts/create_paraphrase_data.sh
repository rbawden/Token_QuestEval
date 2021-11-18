#!/bin/sh

thisdir=`realpath $(dirname $0)`
maindir=$thisdir/..
echo "Main project directory = $maindir"

# Process Parabank 1

# get initial examples - Parabank 1
if [ ! -f $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.tsv.gz ]; then
    echo "Extracting examples to $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.tsv.gz..."
    python  $maindir/scripts/get_one_per_example.py $maindir/data/paraphrase/parabank1.tsv \
	    $maindir/data/paraphrase/parabank1.meta \
	| gzip > $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.tsv.gz
fi
# detokenise Parabank 1
if [ ! -f $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.detok.tsv.gz ]; then
    echo "Detokenising paraphrase data to $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.detok.tsv.gz..."
    paste <(zcat $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.tsv.gz | cut -f 1 | \
		perl $maindir/scripts/dag2txt -l en -nffs -nfc -no_a ) \
	  <(zcat $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.tsv.gz | cut -f 2 | \
		perl $maindir/scripts/dag2txt -l en -nffs -nfc -no_a ) \
	| gzip > $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.detok.tsv.gz
fi
# mask examples Parabank 1
if [ ! -f $maindir/data/paraphrase/parabank1.threshold0.7.detok.masked-examples.jsonl ]; then
    echo "Creating masked examples and outputting to $maindir/data/paraphrase/parabank1.threshold0.7.detok.masked-examples.jsonl..."
    zcat $maindir/data/paraphrase/parabank1.1perexample.threshold0.7.detok.tsv.gz \
	| python -u $maindir/scripts/create_masked_examples.py \
		 > $maindir/data/paraphrase/parabank1.threshold0.7.detok.masked-examples.jsonl
fi
[ -d $maindir/data/paraphrase/parabank1-parts ] || mkdir $maindir/data/paraphrase/parabank1-parts
# split into different parts and rename for consistency Parabank 1
if [ ! -f $maindir/data/paraphrase/parabank1-parts/parabank1.threshold0.7.detok.masked-examples.jsonl.part-1 ]; then
    echo "Splitting data into smaller parts: parabank1.threshold0.7.detok.masked-examples.jsonl.part-{0,1,2...}..."
    split -l 250000 -d -a 3 $maindir/data/paraphrase/parabank1.threshold0.7.detok.masked-examples.jsonl \
	  $maindir/data/paraphrase/parabank1-parts/parabank1.threshold0.7.detok.masked-examples.jsonl.part-
    # rename for consistency
    for file in $maindir/data/paraphrase/parabank1-parts/*part-*; do
	num=`echo $file | perl -pe "s/.+part\-0*(\d+)$/\1/"`
	base=`echo $file | perl -pe 's/part-\d+$/part-/'`
	if [ ! -f $base$num ]; then
	    mv $file $base$num
	fi
    done
fi

# Process Parabank 2

# get initialise examples Parabank 2
if [ ! -f $maindir/data/paraphrase/parabank2.1perexample.tsv.gz ]; then
    echo "Extracting examples to $maindir/data/paraphrase/parabank2.1perexample.tsv.gz..."
    python  $maindir/scripts/get_one_per_example.py $maindir/data/paraphrase/parabank2.tsv None 2 \
	| gzip > $maindir/data/paraphrase/parabank2.1perexample.tsv.gz
fi
# mask examples Parabank 2
if [ ! -f $maindir/data/paraphrase/parabank2.masked-examples.jsonl ]; then
    echo "Creating masked examples and outputting to $maindir/data/paraphrase/parabank2.masked-examples.jsonl..."
    zcat $maindir/data/paraphrase/parabank2.1perexample.tsv.gz \
	| python -u $maindir/scripts/create_masked_examples.py \
		 > $maindir/data/paraphrase/parabank2.masked-examples.jsonl
fi
[ -d $maindir/data/paraphrase/parabank2-parts ] || mkdir $maindir/data/paraphrase/parabank2-parts
# split into different parts and rename for consistency Parabank 2
if [ ! -f $maindir/data/paraphrase/parabank2-parts/parabank2.masked-examples.jsonl.part-1 ]; then
    echo "Splitting data into smaller parts: parabank.threshold0.7.detok.masked-examples.jsonl.part-{0,1,2...}..."
    split -l 250000 -d -a 3 $maindir/data/paraphrase/parabank2.masked-examples.jsonl \
          $maindir/data/paraphrase/parabank2-parts/parabank2.masked-examples.jsonl.part-
    # rename for consistency
    for file in $maindir/data/paraphrase/parabank2-parts/*part-*; do
	num=`echo $file | perl -pe "s/.+part\-0*(\d+)$/\1/"`
	base=`echo $file | perl -pe 's/part-\d+$/part-/'`
	if [ ! -f $base$num ]; then
	    mv $file $base$num
	fi
    done
fi
