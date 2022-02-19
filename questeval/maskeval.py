from typing import List, Tuple, Dict, Callable
import os
import json
import numpy as np
import logging
from datasets import load_metric
import spacy
import torch
import hashlib
from questeval import DIR, __version__
from questeval.utils import (
    API_T2T,
)
from datasets import load_metric
from transformers import T5TokenizerFast

def text2hash(string: str) -> str:
    hash_object = hashlib.sha512(string.encode('utf-8'))
    hex_dig = hash_object.hexdigest()

    return hex_dig

class MaskEval:
    def __init__(
        self,
        fill_mask_model_name = "/home/mila/y/yu-lu.liu/Token_QuestEval/models/t5_config_1_50K/checkpoint-15000",
        language: str = "en",
        no_cuda: bool = False,
        use_cache: bool = True,
        model_batch_size: int = 8,
        sliding_window_size: int=24,
        mask_types_to_consider = ("hyp", "ref",),
        answer_sim_metrics = ("exact_match",),
        order: str = "hyp_first",
    ) -> None:

        #NEEDED
        self.sep = "<sep>"
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'
        self.model_batch_size = model_batch_size
        self.order = order
        self.sliding_window_size = sliding_window_size
        self.fill_mask_model = self.get_model(fill_mask_model_name)

        self.language = language
        self.log_dir = os.path.join(DIR, 'logs')
        self.hash_files = set(os.listdir(self.log_dir))
        self.use_cache = use_cache
        self.mask_types_to_consider = mask_types_to_consider
        self.answer_sim_metrics = answer_sim_metrics
        self.bertscore = load_metric("bertscore")
        self.truncate_tokenizer = T5TokenizerFast.from_pretrained('t5-base')

    def get_model(self, model_name: str, ):
        if 't5' in model_name.lower():
            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                max_seq_length=512,
                model_batch_size=self.model_batch_size,
                sliding_window_size=self.sliding_window_size,
                order = self.order,
                device=self.device
            )
        else:
            raise NotImplementedError(f'Model Name ({model_name}) Not Handled: the model name should contain t5 or mt5')

        return model

    def truncate(self, text_pairs):
        truncated_text_pairs = []
        t5_tokenizer = self.truncate_tokenizer

        for text_pair in text_pairs:
            trucated_hyp_ids = t5_tokenizer(text_pair['hypothesis'], max_length=120, truncation=True)['input_ids']
            trucated_hyp = t5_tokenizer.decode(trucated_hyp_ids)

            trucated_src_ids = t5_tokenizer(text_pair['reference'], max_length=380, truncation=True)['input_ids']
            trucated_src = t5_tokenizer.decode(trucated_src_ids)

            truncated_text_pair = {
                "hypothesis": trucated_hyp,
                "reference": trucated_src,
                "hyp_lang": text_pair["hyp_lang"],
                "ref_lang": text_pair["ref_lang"]
            }

            truncated_text_pairs.append(truncated_text_pair)

        return truncated_text_pairs

    def corpus_questeval(
            self,
            hypothesis: List[str],
            references: List[str],
            batch_size: int = 86
    ) -> Dict:

        assert hypothesis is not None
        assert references is not None
        assert len(references) == len(hypothesis)

        scores = []
        all_logs = []
        for ex_idx in range(0, len(hypothesis), batch_size):
            logging.info(f"Total examples: {len(hypothesis)}. Proceeding the examples {ex_idx}")

            batch_hypothesis = hypothesis[ex_idx:ex_idx + batch_size]
            batch_references = references[ex_idx:ex_idx + batch_size]
            batch_text_pairs = []
            for k in range(len(batch_hypothesis)):
                text_pair = {
                    "hypothesis": batch_hypothesis[k],
                    "reference": batch_references[k],
                    "hyp_lang": self.language,
                    "ref_lang": self.language
                }
                batch_text_pairs.append(text_pair)
            new_scores, logs = self._batch_questeval(
                text_pairs=batch_text_pairs
            )
            scores += new_scores
            all_logs += logs
            

        result = {'corpus_score': np.average(scores), 'ex_level_scores': scores}
        return result, logs

    def _batch_questeval(
            self,
            text_pairs: List[Dict],
    ) -> List[float]:

        d_loaded_logs = dict()
        #text_pairs = self.truncate(text_pairs)
        logs, logs_hashes = self._load_logs(text_pairs, d_loaded_logs)
        self._serialize_logs(logs, logs_hashes)

        #compute the prediction
        do_prediction = False
        for log in logs:
            if not log["prediction_done"]:
                do_prediction = True
                break

        if do_prediction:
            outputs = self.fill_mask_model.predict(text_pairs)
            for k in range(len(logs)):
                log = logs[k]
                log["masked"] = outputs[k]
                log["prediction_done"] = True
            self._serialize_logs(logs, logs_hashes)

        # Compute answer similarity (exact match, BERTScore, etc.)
        do_answ_sim = True #TO SET TO FALSE
        for log in logs:
            if not log["answ_sim_computed"]:
                do_answ_sim = True
                break

        if do_answ_sim:
            self._compute_answer_similarity(logs)
            self._serialize_logs(logs, logs_hashes)

        # Calculate Score
        scores = self._calculate_score_from_logs(logs)
        return scores, logs

    def _compute_answer_similarity(self, logs):
        for log in logs:
            predictions = [log['masked'][w]['prediction'] for w in range(len(log['masked']))]
            gold_labels = [log['masked'][w]['ground_truth'] for w in range(len(log['masked']))]

            #exact match
            em_scores = self._exact_match(predictions, gold_labels)
            for k in range(len(log["masked"])):
                log["masked"][k]["prediction_eval_metrics"]["exact_match"] = em_scores[k]

            #segm-BERTScore
            segm_bertscores = self._segm_bertscore(predictions, gold_labels)
            for k in range(len(log["masked"])):
                log['masked'][k]['seg_bertscore'] = {}
                for typescore in 'precision', 'recall', 'f1':
                    log['masked'][k]['seg_bertscore'][typescore] = segm_bertscores[typescore][k]

            #doc-BERTSCORE
            doc_bertscores = self._doc_berstcore(log)
            for k in range(len(log["masked"])):
                log['masked'][k]['doc_bertscore'] = {}
                for typescore in 'precision', 'recall', 'f1':
                    log['masked'][k]['doc_bertscore'][typescore] = doc_bertscores[typescore][k]

            log["answ_sim_computed"] = True

    def _calculate_bertscore(self, candidates, references, lang='en'):
        assert lang in ['en', 'multilingual']
        if lang == 'en':
            scores = self.bertscore._compute(candidates, references, model_type='bert-base-uncased')
        else:
            scores = self.bertscore._compute(candidates, references, model_type='bert-base-multilingual-cased')
        return scores

    def _exact_match(self, predictions, ground_truths):
        scores = []
        for (p, g) in zip(predictions, ground_truths):
            if p.lower() == g.lower():
                scores.append(1)
            else:
                scores.append(0)
        return scores
    
    def _segm_bertscore(self, predictions, ground_truths):
        return self._calculate_bertscore(pred_labels, gold_labels)

    def _doc_berstcore(self, log):
        original_texts = []
        new_texts = []

        #hypothesis
        hyp_ground_truth_words = [m["ground_truth"] for m in log["masked"] if m["masking"] == "hyp"]
        hyp_predicted_words = [m["prediction"] for m in log["masked"] if m["masking"] == "hyp"]
        for i in range(len(hyp_predicted_words)):
            #construct new sequence with the ground truth word replaced by the prediction
            new_seq = hyp_ground_truth_words.copy()
            new_seq[i] = hyp_predicted_words[i]
            new_text = ' '.join(new_seq)
            new_texts.append(new_text)

        hyp_text = log["hyp_text"]
        original_texts = len(hyp_ground_truth_words) * [hyp_text]

        # reference
        ref_ground_truth_words = [m["ground_truth"] for m in log["masked"] if m["masking"] == "ref"]
        ref_predicted_words = [m["prediction"] for m in log["masked"] if m["masking"] == "ref"]
        for i in range(len(ref_predicted_words)):
            # construct new sequence with the ground truth word replaced by the prediction
            new_seq = ref_ground_truth_words.copy()
            new_seq[i] = ref_predicted_words[i]
            new_text = ' '.join(new_seq)
            new_texts.append(new_text)

        ref_text = log["ref_text"]
        original_texts = len(ref_ground_truth_words) * [hyp_text]

        assert len(original_texts) == len(new_texts)

        return self._calculate_bertscore(new_texts, original_texts)


    def _calculate_score_from_logs(self, logs):
        scores = []
        for log in logs:
            if "hyp" in self.mask_types_to_consider and "ref" in self.mask_types_to_consider:
                mask_hyp_score = self._base_score(log, masking_type = "hyp")
                mask_ref_score = self._base_score(log, masking_type = "ref")
                score = np.average([mask_hyp_score, mask_ref_score])
            elif "hyp" in self.mask_types_to_consider:
                score = self._base_score(log, masking_type="hyp")
            elif "ref" in self.mask_types_to_consider:
                score = self._base_score(log, masking_type="ref")
            else:
                raise("Specify at least one mask_type to consider when computing scores.")

            scores.append(score)

        return scores

    def _base_score(self, log, masking_type):
        metric_scores = []
        for metric in self.answer_sim_metrics:
            for l in log["masked"]:
                if l["masking"] == masking_type:
                    if metric in l["prediction_eval_metrics"]:
                        metric_scores.append(l["prediction_eval_metrics"][metric])
                    else:
                        logging.warning(f"answer similarity metric {metric} is not in the logs. Setting the metric score to 0. ")
                        metric_scores.append(0)

        return np.average(metric_scores)

    def _serialize_logs(
        self,
        logs: List[Dict],
        hashes: List[str]
    ) -> None:
        for log, hash in zip(logs, hashes):
            with open(os.path.join(self.log_dir, hash), 'w',  encoding="utf-8") as outfile:
                json.dump(log, outfile, indent=2)

    def open_log_from_text(self, hypothesis: str, reference: str) -> Dict:
        id_text = hypothesis + self.sep + reference
        log_hash = text2hash(id_text)
        with open(os.path.join(self.log_dir, log_hash), 'r', encoding="utf-8") as f_log:
            log = json.load(f_log)
        return log

    def _load_logs(
        self,
        text_pairs: Dict,
        d_loaded_logs: Dict
    ) -> Tuple[List[Dict], List[str]]:
        logs, log_hashs = [], []

        for text_pair in text_pairs:
            hyp_text = text_pair["hypothesis"]
            ref_text = text_pair["reference"]
            id_text =  hyp_text + self.sep + ref_text

            log_hash = text2hash(id_text)
            if log_hash not in d_loaded_logs:
                log = {'id_text': id_text,
                       'hyp_text': hyp_text,
                       'ref_text': ref_text,
                       'prediction_done': False,
                       'answ_sim_computed': False,
                       'weights_computed': False,
                       'masked': dict()}
                if not (self.use_cache and log_hash in self.hash_files and id_text != ""):
                    temp=1
                if self.use_cache and log_hash in self.hash_files and id_text != "":
                    cached_path = os.path.join(self.log_dir, log_hash)
                    try:
                        with open(cached_path, 'r', encoding="utf-8") as f_log:
                            tmp  = json.load(f_log)
                            assert all([k in log for k in ['id_text', 'hyp_text', 'ref_text', 'masked', 'prediction_done','answ_sim_computed', 'weights_computed']])
                            assert isinstance(log['id_text'], str)
                            assert isinstance(log['hyp_text'], str)
                            assert isinstance(log['ref_text'], str)
                            assert isinstance(log['masked'], dict)
                            assert isinstance(log['prediction_done'], bool)
                            assert isinstance(log['answ_sim_computed'], bool)
                            assert isinstance(log['weights_computed'], bool)
                            log = tmp
                    except json.decoder.JSONDecodeError:
                        self.hash_files.remove(log_hash)
                        os.remove(cached_path)
                    except AssertionError:
                        self.hash_files.remove(log_hash)
                        os.remove(cached_path)

                d_loaded_logs[log_hash] = log

            logs.append(d_loaded_logs[log_hash])
            log_hashs.append(log_hash)

        return logs, log_hashs
