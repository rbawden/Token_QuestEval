#!/bin/sh

thisdir=`dirname $0`
maindir=`realpath($dirname/..)`
echo "Main project directory = $maindir"

# Parabank v1
if [ ! -d $maindir/data/paraphrase/parabank-full ]; then
    # download full parabank
    wget http://cs.jhu.edu/~vandurme/data/parabank-1.0-full.zip -O $maindir/data/paraphrase/parabank-1.0-full.zip
    # unzip it
    unzip $maindir/data/paraphrase/parabank-1.0-full.zip -d $maindir/data/paraphrase/
    # delete zipped data
    rm $maindir/data/paraphrase/parabank-1.0-full.zip
    # move data out of folder
    mv $maindir/data/paraphrase/full/parabank.tsv $maindir/data/paraphrase/parabank1
done

# Parabank v2
if [ ! -d $maindir/data/paraphrase/parabank2 ]; then
    # download full parabank
    wget http://cs.jhu.edu/~vandurme/data/parabank-2.0 -O $maindir/data/paraphrase/parabank-2.0.zip
    # unzip it
    unzip $maindir/data/paraphrase/parabank-2.0.zip -d $maindir/data/paraphrase/
    # delete zipped data
    rm $maindir/data/paraphrase/parabank-2.0.zip
done

# WMT metrics task submission data

# WMT metrics task data
