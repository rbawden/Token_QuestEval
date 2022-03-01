#!/usr/bin/python
import json
from numpy import mean

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


def str2func(string):
    if string == 'mean':
        return mean
    elif string == 'min':
        return min
    elif string == 'max':
        return max


def read_log(json_file, metric, level='seg', sent_agg_func=mean, word_agg_func=mean):
    total_score = 0
    total = 0
    
    # convert function names to functions
    word_agg_func = str2func(word_agg_func)
    sent_agg_func = str2func(sent_agg_func)

    with open(json_file) as fp:
        for ex_id, example in enumerate(fp):
            ex_log = json.loads(example)

            #print(ex_log)
            #input()
            # get basic scores
            #-----------------
            if metric in ['exact_match_hyp', 'exact_match_both']:
                exact_match_hyp = [word['ground_truth']==word['prediction'] for word in ex_log['masked'] \
                                   if word['masking']=='hyp' if word['ground_truth'] != '']
            if metric in ['exact_match_ref', 'exact_match_both']:    
                exact_match_ref = [word['ground_truth']==word['prediction'] for word in ex_log['masked'] \
                                   if word['masking']=='ref' if word['ground_truth'] != '']
            if metric == 'exact_match_hyp':
                score = sum(exact_match_hyp) / len(exact_match_hyp)
            elif metric == 'exact_match_ref':
                score = sum(exact_match_ref) / len(exact_match_ref)
            elif metric == 'exact_match_both':
                score = sum(exact_match_ref + exact_match_hyp) / len(exact_match_ref + exact_match_hyp)

            if metric in ['pred_scores_hyp', 'pred_scores_both']:
                pred_scores_hyp = [word['prediction_eval_metrics']['pred_scores'] for word in ex_log['masked'] \
                                   if word['masking']=='hyp' if word['ground_truth'] != '']
            if metric in ['pred_scores_ref', 'pred_scores_both']:
                pred_scores_ref = [word['prediction_eval_metrics']['pred_scores'] for word in ex_log['masked'] \
                                   if word['masking']=='ref' if word['ground_truth'] != '']
            if metric in ['gold_scores_hyp', 'gold_scores_both']:
                gold_scores_hyp = [word['prediction_eval_metrics']['gold_scores'] for word in ex_log['masked'] \
                                   if word['masking']=='hyp' if word['ground_truth'] != '']
            if metric in ['gold_scores_ref', 'gold_scores_both']:
                gold_scores_ref = [word['prediction_eval_metrics']['gold_scores'] for word in ex_log['masked'] \
                                   if word['masking']=='ref' if word['ground_truth'] != '']
            
            if metric == 'pred_scores_hyp':
                score = 1 - sent_agg_func([word_agg_func(word_scores) for word_scores in pred_scores_hyp])
            elif metric == 'pred_scores_ref':
                score = 1 - sent_agg_func([word_agg_func(word_scores) for word_scores in pred_scores_ref])
            elif metric == 'pred_scores_both':
                score = 1 - sent_agg_func([word_agg_func(word_scores) for word_scores in pred_scores_hyp + pred_scores_ref])
            if metric == 'gold_scores_hyp':
                score = 1 - sent_agg_func([word_agg_func(word_scores) for word_scores in gold_scores_hyp])
            elif metric == 'gold_scores_ref':
                score = 1 - sent_agg_func([word_agg_func(word_scores) for word_scores in gold_scores_ref])
            elif metric == 'gold_scores_both':
                score = 1 - sent_agg_func([word_agg_func(word_scores) for word_scores in gold_scores_hyp + gold_scores_ref])

            if metric in ['bert_p_hyp', 'bert_p_both']:
                bert_scores_hyp = [word['bert_scores']['precision'] for word in ex_log['masked'] \
                                   if word['masking'] == 'hyp' if word['ground_truth'] != '']
            elif metric in ['bert_p_ref', 'bert_p_both']:
                bert_scores_hyp = [word['bert_scores']['precision'] for word in ex_log['masked'] \
                                   if word['masking'] == 'ref' if word['ground_truth'] != '']
            elif metric in ['bert_r_hyp', 'bert_r_both']:
                bert_scores_hyp = [word['bert_scores']['recall'] for word in ex_log['masked'] \
                                   if word['masking'] == 'hyp' if word['ground_truth'] != '']
            elif metric in ['bert_r_ref', 'bert_r_both']:
                bert_scores_hyp = [word['bert_scores']['recall'] for word in ex_log['masked'] \
                                   if word['masking'] == 'ref' if word['ground_truth'] != '']
            elif metric in ['bert_f_hyp', 'bert_f_both']:
                bert_scores_hyp = [word['bert_scores']['f1'] for word in ex_log['masked'] \
                                   if word['masking'] == 'hyp' if word['ground_truth'] != '']
            elif metric in ['bert_f_ref', 'bert_f_both']:
                bert_scores_hyp = [word['bert_scores']['f1'] for word in ex_log['masked'] \
                                   if word['masking'] == 'ref' if word['ground_truth'] != '']
            
            if metric == 'bert_p_hyp':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_p_hyp])
            elif metric == 'bert_p_ref':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_p_ref])
            elif metric == 'bert_p_both':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_p_ref + bert_p_hyp])

            if metric == 'bert_r_hyp':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_r_hyp])
            elif metric == 'bert_r_ref':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_r_ref])
            elif metric == 'bert_r_both':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_r_ref + bert_r_hyp])

            if metric == 'bert_f_hyp':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_f_hyp])
            elif metric == 'bert_f_ref':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_f_ref])
            elif metric == 'bert_f_both':
                score = sent_agg_func([word_agg_func(word_scores) for word_scores in bert_f_ref + bert_f_hyp])

            #-----------------
            # print out score
            if level == 'seg':
                print(score)
            else:
                total_score += score
                total += 1

            continue

            # for debugging purposes

            for w, word in enumerate(ex_log['masked']):
                if word['ground_truth'] == '':
                    continue
                print(word['ground_truth'])
                if word['bert_scores']['precision'] == 0:
                    #print(ex_log)
                    print(word)
                    print(w)
                    input()
    if level == 'sys':
        print(total_score / total)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_log')
    parser.add_argument('type_score', choices=('exact_match_both', 'exact_match_hyp', 'exact_match_ref',
                                               'pred_scores_both', 'pred_scores_hyp', 'pred_scores_ref',
                                               'gold_scores_both', 'gold_scores_hyp', 'gold_scores_ref', 
                                               'bert_p_ref','bert_p_hyp', 'bert_p_both'
                                               'bert_r_ref','bert_r_hyp', 'bert_r_both'
                                               'bert_f_ref','bert_f_hyp', 'bert_f_both'
                                           )
    )
    parser.add_argument('-l', '--level', default='seg', choices=('sys', 'seg'))
    parser.add_argument('--sent_agg', default='mean', choices=('mean', 'max', 'min'))
    parser.add_argument('--word_agg', default='mean', choices=('mean', 'max', 'min'))
    args = parser.parse_args()
    
    read_log(args.json_log, args.type_score, level=args.level, sent_agg_func=args.sent_agg, word_agg_func=args.word_agg)
