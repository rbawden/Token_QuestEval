#!/usr/bin/python
from transformers import T5Tokenizer, T5ForConditionalGeneration
from questeval.maskeval import MaskEval
import torch
import json
import numpy as np

def p_r_f_score(hyp, ref):
    hyp_toks = hyp.split() # other tokenisation?
    ref_toks = ref.split() # ditto
    same = [x for x in hyp_toks if x in ref_toks]
    p = same / len(hyp_toks)
    r = same / len(ref_toks)
    return

def bert_score(hyps, refs):
    bert_p, bert_r, bert_f = bert_score.score(hyps, refs, lang='en', rescale_with_baseline=True)
    return bert_p, bert_r, bert_f


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


def new_predict(model_path, hyp, source):
    maskeval = MaskEval(fill_mask_model_name = model_path,
                        use_cache=False)
    hyps = []
    refs = []
    with open(hyp) as hf, open(source) as sf:
        for sid, (h, s) in enumerate(zip(hf, sf)):
            hyps.append(h)
            refs.append(s)
            if len(hyps) == 20:
                break
    score, logs = maskeval.corpus_questeval(hypothesis=hyps, references=refs)

    # make sure in correct order

    print('exact\tpredmean\tminpred\tmaxpred')
    for l, log in enumerate(logs):
        exact_match_score = aggregate(log['masked'], 'exact_match')
        mean_pred_score = aggregate(log['masked'], 'pred_scores', np.mean)
        min_pred_score = aggregate(log['masked'], 'pred_scores', np.min)
        max_pred_score = aggregate(log['masked'], 'pred_scores', np.max)
        
        example = [round(exact_match_score, 4), round(mean_pred_score, 4),
                   round(min_pred_score, 4), round(max_pred_score, 4)]
        print('\t'.join([str(x) for x in example]))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('hyp')
    parser.add_argument('source')
    args = parser.parse_args()

    new_predict(args.model, args.hyp, args.source)
