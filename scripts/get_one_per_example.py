#!/usr/bin/python
import random
import json
import os

def process_version1(full_dataset, meta_dataset):
    with open(full_dataset) as fp, open(meta_dataset) as dfp:
        for line, meta_str in zip(fp, dfp):
            examples = line.strip().split('\t')
            meta = json.loads(meta_str)
            first = examples[0]
            others = examples[1:]
 
            # filter poor examples
            filtered = []
 
            for example, ex_meta in zip(others, meta['paraphrases']):
                if example == first or float(ex_meta['model_score']) < 0.7:
                    continue
                filtered.append(example)
            if len(filtered) > 0:
                #random.shuffle(filtered)
                second = filtered[0]
                print('\t'.join([second, first]))
            
def process_version2(full_dataset):
    with open(full_dataset) as fp:
        for line in fp:
            line_tab = line.strip().split('\t')
            if len(line_tab) < 3:
                continue
            score, first = line_tab[0], line_tab[1]
            examples = line_tab[2:]
            # take a random example
            random.shuffle(examples)
            second = examples[0]
            print('\t'.join([second, first]))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('full_dataset')
    parser.add_argument('meta_scores')
    parser.add_argument('version', choices=(1, 2), type=int)
    args = parser.parse_args()

    if args.version == 1:
        process_version1(args.full_dataset, args.meta_scores)
    else:
        process_version2(args.full_dataset)
