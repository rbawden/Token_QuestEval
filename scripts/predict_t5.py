#!/usr/bin/python
from transformers import T5Tokenizer, T5ForConditionalGeneration
from questeval.maskeval import MaskEval
import torch
import json
import numpy as np

def aggregate(words, metric_name, func=None):
    score = 0
    num_words = len(words)
    for word in words:
        if metric_name in ['exact_match']:
            score += word['comparison_metrics'][metric_name] / len(words)
        elif metric_name in ['pred_scores']:
            assert func is not None, 'A function must be defined with this metric: ' + metric_name
            score += func(word['pred_scores'])
    return score


def predict(model_path, hyp, source):
    maskeval = MaskEval(fill_mask_model_name = model_path,
                        use_cache=False)
    # read hypothesis and reference files
    hyps, refs = [], []
    with open(hyp) as hf, open(source) as sf:
        for sid, (h, s) in enumerate(zip(hf, sf)):
            hyps.append(h)
            refs.append(s)
            if len(hyps) == 5:
                break

    # make predictions and get scores (stored in log files)
    _, logs = maskeval.corpus_questeval(hypothesis=hyps, references=refs)

    # go through log files and print out all scores for each example
    headers, printed_headers = '', False
    for l, log in enumerate(logs):
        example = []
        print(log)

        # exact match scores (averaged over all words)
        example.append(round(aggregate(log['masked'], 'exact_match'), 4))
        headers += 'exact'

        # logit scores of predicted words and scores of gold words
        for name_score in 'pred_scores', 'gold_scores':
            # different aggregation functions tested
            for agg_func in np.mean, np.min, np.max:
                example.append(round(aggregate(log['masked'], name_score, agg_func), 4))
                headers += '\t' + name_score + '_' + agg_func.__name__

        # bert-score aggregate scores comparing each predicted word with its ground truth
        headers += 'bertscore_labels'
        # different aggregation functions tested
        for agg_func in np.mean, np.min, np.max:
            for berttype in 'precision', 'recall', 'fscore':
                all_scores = [w['bertscore'][berttype] for w in log['masked']]
                example.append(agg_func(all_scores))
                headers += '\tbertscore_' + agg_func.__name__
        # bert-score on whole predicted sequences (i.e. if we take each masked predictions and treat it as a sequence
        # to beb compared to the other sequence (tested this out)
        for berttype in 'bertscore_hyp_mlmpred', 'bertscore_ref_mlmpred', 'bertscore_ref_hyp':
            example.append(log['comparison_metrics'][berttype])
            headers += '\t' + berttype
        
        # print headers first time
        if not printed_headers:
            print(headers.strip())
            printed_headers = True
        # print example scores
        print('\t'.join([str(x) for x in example]))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('hyp')
    parser.add_argument('source')
    args = parser.parse_args()

    predict(args.model, args.hyp, args.source)
