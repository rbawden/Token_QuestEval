#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath($dirname/..)`
echo "Main project directory = $maindir"

# get initial examples - Parabank v 1
for version in 1 2; do
    if [ ! -f $maindir/data/paraphrase/parabank$version.1perexample.threshold0.7.tsv.gz ]; then
	echo "Extracting examples to $maindir/data/paraphrase/parabank$version.1perexample.threshold0.7.tsv.gz..."
	python  $maindir/data/paraphrase/get_one_per_example.py $maindir/data/paraphrase/parabank$version.tsv \
		$maindir/data/paraphrase/full/parabank.meta | gzip > $maindir/data/paraphrase/parabank$version.1perexample.threshold0.7.tsv.gz
    fi
done
if [ ! -f $maindir/data/paraphrase/parabank2.1perexample.threshold0.7.tsv.gz ]; then
    # get initialise examples Parabank v 2
    echo "Extracting examples to $maindir/data/paraphrase/parabank.1perexample.threshold0.7.tsv.gz..."
    python  $maindir/data/paraphrase/get_one_per_example.py $maindir/data/paraphrase/parabank2.tsv \
            $maindir/data/paraphrase/full/parabank.meta | gzip > $maindir/data/paraphrase/parabank2.1perexample.threshold0.7.tsv.gz
fi
# detokenise
if [ ! -f $maindir/data/paraphrase/parabank.1perexample.threshold0.7.detok.tsv.gz ]; then
    echo "Detokenising paraphrase data to $maindir/data/paraphrase/parabank.1perexample.threshold0.7.detok.tsv.gz..."
    paste <(zcat $maindir/data/paraphrase/parabank.1perexample.threshold0.7.tsv.gz | cut -f 1 | perl scripts/dag2txt -l en -nffs -nfc -no_a ) \
	  <(zcat $maindir/data/paraphrase/parabank.1perexample.threshold0.7.tsv.gz | cut -f 2 | perl scripts/dag2txt -l en -nffs -nfc -no_a ) \
	| gzip > $maindir/data/paraphrase/parabank.1perexample.threshold0.7.detok.tsv.gz
fi
# mask examples
if [ ! -f $maindir/data/paraphrase/parabank.threshold0.7.detok.masked-examples.jsonl ]; then
    echo "Creating masked examples and outputting to $maindir/data/paraphrase/parabank.threshold0.7.detok.masked-examples.jsonl..."
    zcat $maindir/data/paraphrase/parabank.1perexample.threshold0.7.detok.tsv.gz \
	| python -u scripts/create_masked_examples.py \
		 > $maindir/data/paraphrase/parabank.threshold0.7.detok.masked-examples.jsonl
fi
# split into different parts and rename for consistency
if [ ! -f parabank.threshold0.7.detok.masked-examples.jsonl.part-1 ]; then
    echo "Splitting data into smaller parts: parabank.threshold0.7.detok.masked-examples.jsonl.part-{0,1,2...}..."
    split -l 20000 -d -a 3 $maindir/data/paraphrase/parabank.threshold0.7.detok.masked-examples.jsonl \
	  parabank.threshold0.7.detok.masked-examples.jsonl.part-
fi
