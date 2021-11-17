#!/usr/bin/python
import random
import json
#from mosestokenizer import MosesDetokenizer

def process(full_dataset, meta_dataset):
    #detok = MosesDetokenizer('en')
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
                print('\t'.join([first, second]))
            


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('full_dataset')
    parser.add_argument('meta_scores')
    args = parser.parse_args()

    process(args.full_dataset, args.meta_scores)
