#!/usr/bin/env python3

"""
This converts files of raw complete segment scores for a set of systems
into the WMT "agree format".

we need this format:

    SID BETTER WORSE chrF sentBLEU CharacTER BEER ITER RUSE chrF+ meteor++ YiSi-1 YiSi-0 BLEND YiSi-1_srl UHH_TSKM
    344 CUNI-Transformer.5560 online-B.0 1 1 1 1 1 1 1 1 1 1 1 1 1
    345 online-G.0 CUNI-Transformer.5560 0 0 0 0 0 0 0 0 0 0 0 0 0

and we have system files, with a score per line, all in a directory, ending in *.seg, like this:

    0.1
    0.2
    ...
"""

import argparse
import sys
import csv
import glob
import os.path

from csv import DictReader, DictWriter
from collections import defaultdict

def read_agree(agree_file):
    gold = {}
    for row in DictReader(agree_file):
        sid = int(row["SID"])
        if sid not in gold:
            gold[sid] = []
        gold[sid].append((row["BETTER"], row["WORSE"]))
    return gold


def normalize(sysname):
    return sysname.replace("_", "-")


def main(args):
    # Read in the scores from all systems found in the system directory
    scores = {}
    #print(args.scores_dir)
    for system_file in glob.glob(f"{args.scores_dir}/*.seg"):
        name = normalize(".".join(os.path.basename(system_file).split(".")[0:-1]))
        # shouldn't normally need this unless debugging
        if name == 'cache':
            continue
        with open(system_file) as infile:
            scores[name] = list(map(lambda x: float(x.rstrip()), infile.readlines()))
            #print(name, len(scores[name]), scores[name][:10], file=sys.stderr)

    # Now go through the official results file, line by line, computing agreement
    correct = 0
    incorrect = 0
    for row in DictReader(args.agree_file, delimiter=" "):
        sentno = int(row["SID"]) - 1
        better = normalize(row["BETTER"])
        worse = normalize(row["WORSE"])
        try:
            agreed = scores[better][sentno] > scores[worse][sentno]
        except:
            print(f"No key in {args.agree_file.name} for sentence {sentno} better={better} or worse={worse}", file=sys.stderr)
            print("ERR")
            sys.exit(0)
        if agreed:
            correct += 1
        else:
            incorrect += 1

        row = {
            "SID": row["SID"],
            "BETTER": row["BETTER"],
            "WORSE": row["WORSE"],
            args.name: int(agreed),
        }

    print(f"{(correct - incorrect) / (correct + incorrect):.3f}")

    sys.exit(1)
    scores = defaultdict(dict)
    # baseline-540000 cs-en   newstest2018    CUNI-Transformer.5560   1       19.13
    for line in open(args.ourfile):
        _, langpair, _, system, sentno, score = line.rstrip().split('\t')

        key = (langpair.replace('-', ''), int(sentno))
        scores[key][system] = float(score)


    print(f'Will load from {args.agree_file}', file=sys.stderr)

    # SID BETTER WORSE chrF sentBLEU CharacTER BEER ITER RUSE chrF+ meteor++ YiSi-1 YiSi-0 BLEND YiSi-1_srl UHH_TSKM
    # 344 CUNI-Transformer.5560 online-B.0 1 1 1 1 1 1 1 1 1 1 1 1 1
    # 345 online-G.0 CUNI-Transformer.5560 0 0 0 0 0 0 0 0 0 0 0 0 0
    out = open('out', 'w')
    metric = os.path.basename(args.ourfile)
    print(f'SID BETTER WORSE {metric}', file=out)
    d = csv.DictReader(open(args.agree_file), delimiter=' ')
    skipped = 0
    total = 0
    for row in d:
        total += 1
        sentno = int(row['SID'])
        better = row['BETTER']
        worse = row['WORSE']
        key = (args.langpair,sentno)
    #    print(key, better, worse, scores[key][better], scores[key][worse])
        if key not in scores or better not in scores[key] or worse not in scores[key]:
    #        print('FAIL', key, better, worse, key not in scores, better not in scores[key], file=sys.stderr)
    #        print(better, scores[key], file=sys.stderr)
            skipped += 1
            continue
        agree = 1 if scores[key][better] > scores[key][worse] else 0
        print(f'{sentno} {better} {worse} {agree}', file=out)
    out.close()

    score = score_csv(open('out'), metric)
    print(f"Skipped {skipped}/{total} entries that weren't found", file=sys.stderr)
    print(f'{score:.8f}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("agree_file", type=argparse.FileType("r"), help="gold agree file")
    parser.add_argument("scores_dir", help="directory with system scores")
    parser.add_argument("name", help="run to score; should find {scores_dir}/*.{name}.each")
    args = parser.parse_args()

    main(args)
