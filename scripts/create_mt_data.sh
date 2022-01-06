#!/bin/sh

thisdir=`realpath $(dirname $0)`
maindir=$thisdir/..
echo "Main project directory = $maindir"


bash $maindir/scripts/concatenate_wmt_data.sh | \
    perl $maindir/scripts/normalize-punctuation.perl -l en | gzip > $maindir/data/metrics/wmt14-18-intoEnglish-all.hyp-ref.tsv.gz

zcat $maindir/data/metrics/wmt14-18-intoEnglish-all.hyp-ref.tsv.gz | \
    python -u $maindir/scripts/create_dataset.py --hyp_lang 'en' --ref_lang 'en' \
           > $maindir/data/metrics/wmt14-18-intoEnglish-all.hyp-ref.jsonl
