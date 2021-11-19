#!/bin/sh

thisdir=`realpath $(dirname $0)`
maindir=$thisdir/..
echo "Main project directory = $maindir"


bash $maindir/scripts/concatenate_wmt_data.sh.sh $maindir/data/metrics/ | \
    perl $maindir/scripts/normalize-punctuation.perl -l en | gzip > $maindir/data/metrics/wmt14-18-intoEnglish-all.hyp-ref.tsv.gz

zcat $maindir/data/metrics/wmt14-18-intoEnglish-all.hyp-ref.tsv.gz | \
    python -u $maindir/scripts/create_masked_examples.py data/metrics/wmt14-18-intoEnglish-all.hyp-ref.tsv \
           > $maindir/data/metrics/wmt14-18-intoEnglish-all.hyp-ref.masked-examples.jsonl
