import torch
import json
import logging
import os
import random

def load_jsonl(input_path) -> list:
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data

def aggregate_expert_scores(expert_annotations):
    """
    expert_annotations is a list of dictionaries:
      "expert_annotations": [
                {
                  "coherence": 5,
                  "consistency": 5,
                  "fluency": 5,
                  "relevance": 5
                },
                {
                  "coherence": 4,
                  "consistency": 5,
                  "fluency": 5,
                  "relevance": 5
                },
                {
                  "coherence": 4,
                  "consistency": 5,
                  "fluency": 5,
                  "relevance": 4
                }
          ]
    where each dictionary correspond to the evaluation scores given by an expert annotator
    """
    #coherence_list = [expert["coherence"] for expert in expert_annotations]
    consistency_list = [expert["consistency"] for expert in expert_annotations]
    #fluency_list = [expert["fluency"] for expert in expert_annotations]
    #relevance_list = [expert["relevance"] for expert in expert_annotations]

    #avg_coherence = sum(coherence_list)/len(coherence_list)
    avg_consistency = sum(consistency_list) / len(consistency_list)
    #avg_fluency = sum(fluency_list) / len(fluency_list)
    #avg_relevance = sum(relevance_list) / len(relevance_list)

    #scale so the aggregated score is in the same numerical range as MaskEval score without learned weights (0-1)
    agg_score = avg_consistency/5
    return agg_score

def get_ids(log, masking):
    """
    log["masked"] is a list of dictionaries, each dictionary corresponds to a single masked sequence like so:
    {
      "prediction": "Italy",
      "ground_truth": "America",
      "ground_truth_ids": [1371],
      "masking": "ref",
      "prediction_eval_metrics": {
        "pred_scores": [-1.2320034503936768],
        "gold_scores": [-34.40394973754883],
        "exact_match": 0,
        "seg_bertscore": {"precision": 0.6720162630081177, "recall": 0.6720162630081177, "f1": 0.6720162630081177},
      }

    masking is a string: either "ref" for reference/source document, or "hyp" for hypothesis
    get_ids returns a list. Each element of the list being a list of token corresponding to a masked sequence.
    only masked sequences from {masking} type are taken into account.
    """
    id_lst = []
    for m in log["masked"]:
        if m["masking"] == masking:
            id_lst.append(m["ground_truth_ids"])
    return id_lst

def get_scores(log, masking, score_type = "exact_match"):
    """
    FOR NOW ONLY SUPPORTS EXACT MATCH ==> score_type = "exact_match"
    log["masked"] is a list of dictionaries, each dictionary corresponds to a single masked sequence like so:
    {
      "prediction": "Italy",
      "ground_truth": "America",
      "ground_truth_ids": [1371],
      "masking": "ref",
      "prediction_eval_metrics": {
        "pred_scores": [-1.2320034503936768],
        "gold_scores": [-34.40394973754883],
        "exact_match": 0,
        "seg_bertscore": {"precision": 0.6720162630081177, "recall": 0.6720162630081177, "f1": 0.6720162630081177},
      }

    masking is a string: either "ref" for reference/source document, or "hyp" for hypothesis
    get_scores returns a list. Each element of the list being a (numeric) prediction eval score {score_type} of a masked sequence.
    only masked sequences from {masking} type are taken into account
    """
    scores_lst = []
    for m in log["masked"]:
        if m["masking"] == masking:
            metrics = m["prediction_eval_metrics"]
        if score_type in metrics:
            scores_lst.append(metrics[score_type])
        else:
            raise ValueError('No such score_type in the log')
    return scores_lst

def process_log(log, score_type, mode):
    """
    Takes all the necessary information from a log file, for either training or inference.
    """

    #get the list of lists of IDs
    hyp_word_list = get_ids(log, "hyp")
    ref_word_list = get_ids(log, "ref")

    if mode == "train":
        hyp_scores = get_scores(log, "hyp", score_type)
        ref_scores = get_scores(log, "ref", score_type)
        expert_score = aggregate_expert_scores(log["expert_annotations"])

        #assert that the lengths match
        assert len(hyp_word_list) == len(hyp_scores)
        assert len(ref_word_list) == len(ref_scores)

        #expand the scores lists
        #example "don't" has exact match = 1 (fill-mask predicted exactly "don't")
        #example "sleep" has exact match = 0 (fill-mask predicted something like "eat" for example)
        #word_list = [["do", "n", "'", "t"], ["sleep"]] (in string form for illustration, they are token IDs in reality)
        #expanded_scores = [1, 1, 1, 1, 0]
        expanded_hyp_scores = []
        for i in range(len(hyp_word_list)):
            expanded_hyp_scores += (len(hyp_word_list[i])*[hyp_scores[i]])

        expanded_ref_scores = []
        for i in range(len(ref_word_list)):
            expanded_ref_scores += (len(ref_word_list[i])*[ref_scores[i]])

    if mode == "train":
        return{
          "hyp_word_list": hyp_word_list,
          "ref_word_list": ref_word_list,
          "hyp_scores": expanded_hyp_scores,
          "ref_scores": expanded_ref_scores,
          "gold_score": expert_score
        }
    else:
        return {
          "hyp_word_list": hyp_word_list,
          "ref_word_list": ref_word_list,
        }
