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
from questeval.emb_weighter import Emb_Weighter
from datasets import load_metric

def text2hash(string: str) -> str:
    hash_object = hashlib.sha512(string.encode('utf-8'))
    hex_dig = hash_object.hexdigest()

    return hex_dig

class MaskEval:
    def __init__(
        self,
        fill_mask_model_name = "/home/mila/y/yu-lu.liu/Token_QuestEval/models/mt5_hyp_first_all/checkpoint-57714",
        use_weighter: bool = False,
        use_sparse_attn: bool = False,
        attn_threshold = 0.10,
        weighter_path = "/home/mila/y/yu-lu.liu/Token_QuestEval/questeval/coh_emb_base_weighter.pt-48",
        no_cuda: bool = False,
        use_cache: bool = True,
        model_batch_size: int = 8,
        sliding_window_size: int=24,
        mask_types_to_consider = ("hyp", "ref",),
        answer_sim_metrics = ("exact_match",),
        order: str = "hyp_first",
    ) -> None:

        self.sep = "<sep>"
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self.use_weighter = use_weighter
        self.weighter_path = weighter_path
        self.weighter = Emb_Weighter()
        if self.device == "cuda":
            self.weighter.cuda()
        self.weighter.load_state_dict(torch.load(self.weighter_path))
        self.weighter.eval()

        self.use_sparse_attn = use_sparse_attn
        self.attn_threshold =  attn_threshold

        self.model_batch_size = model_batch_size
        self.order = order
        self.sliding_window_size = sliding_window_size
        self.fill_mask_model_name = fill_mask_model_name
        self.fill_mask_model = self.get_model(fill_mask_model_name)
        self.log_dir = os.path.join(DIR, 'logs')
        self.hash_files = set(os.listdir(self.log_dir))
        self.use_cache = use_cache
        self.mask_types_to_consider = mask_types_to_consider
        self.answer_sim_metrics = answer_sim_metrics
        self.bertscore = load_metric("bertscore")

        self.attn_counts = {
            "hyp": [],
            "ref": [],
        }

        self.summeval_lang = {
            'DE': "de",
            'ES': "es",
            'FR': "fr",
            'RU': "ru",
            'TR': "tu",
            'EN': "en",
            'en': "en",
            'ZH': "zh",
            'ID': "id"
        }

    def get_model(self, model_name: str, ):
        if 't5' in model_name.lower():
            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                max_seq_length=512,
                model_batch_size=self.model_batch_size,
                sliding_window_size=self.sliding_window_size,
                order = self.order,
                device = self.device
            )
        else:
            raise NotImplementedError(f'Model Name ({model_name}) Not Handled: the model name should contain t5 or mt5')

        return model

    def corpus_questeval(
            self,
            hypothesis: List[str],
            references: List[str],
            languages: List[str],
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
            batch_languages = languages[ex_idx:ex_idx + batch_size]
            batch_text_pairs = []
            for k in range(len(batch_hypothesis)):
                text_pair = {
                    "hypothesis": batch_hypothesis[k],
                    "reference": batch_references[k],
                    "hyp_lang": self.summeval_lang[batch_languages[k]],
                    "ref_lang": self.summeval_lang[batch_languages[k]]
                }
                batch_text_pairs.append(text_pair)
            new_scores, logs = self._batch_questeval(
                text_pairs=batch_text_pairs
            )
            scores += new_scores
            all_logs += logs
            

        result = {'corpus_score': np.average(scores), 'ex_level_scores': scores}
        if self.mask_types_to_consider == ("hyp", "ref",) and self.use_sparse_attn:
            print(sum(self.attn_counts["ref"])/len(self.attn_counts["ref"]))
            print(sum(self.attn_counts["hyp"])/len(self.attn_counts["hyp"]))
        return result #, all_logs

    def _batch_questeval(
            self,
            text_pairs: List[Dict],
    ) -> List[float]:

        d_loaded_logs = dict()
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
                log["fill_mask_model_name"] = self.fill_mask_model_name
            self._serialize_logs(logs, logs_hashes)

        # Compute answer similarity (exact match, BERTScore, etc.)
        do_answ_sim = False
        for log in logs:
            if not log["answ_sim_computed"]:
                do_answ_sim = True
                break

        if do_answ_sim:
            self._compute_answer_similarity(logs)
            self._serialize_logs(logs, logs_hashes)

        # Compute weights
        compute_weighter = False
        for log in logs:
            if not log["weights_computed"]:
                compute_weighter = False
                break

        if compute_weighter:
            self._compute_weights(logs)
            self._serialize_logs(logs, logs_hashes)

        # Calculate Score
        scores = self._calculate_score_from_logs(logs)
        return scores, logs

    def _compute_weights(self, logs):
        for log in logs:
            weighter_outputs = self.weighter.compute_weights(log)

            for i in range(len(log["masked"])):
                log["masked"][i]["weight"] = {
                    "weight_values": weighter_outputs[i]["weight"],
                    "scores": weighter_outputs[i]["score"],
                }

            log["weights_computed"] = True
            log["weighter_name"] = self.weighter_path

    def _compute_answer_similarity(self, logs):
        for log in logs:
            predictions = [log['masked'][w]['prediction'] for w in range(len(log['masked']))]
            gold_labels = [log['masked'][w]['ground_truth'] for w in range(len(log['masked']))]

            #exact match
            em_scores = self._exact_match(predictions, gold_labels)
            for k in range(len(log["masked"])):
                log["masked"][k]["prediction_eval_metrics"]["exact_match"] = em_scores[k]

            """
            #segm-BERTScore
            segm_bertscores = self._segm_bertscore(predictions, gold_labels)
            for k in range(len(log["masked"])):
                log['masked'][k]["prediction_eval_metrics"]['seg_bertscore'] = {}
                for typescore in 'precision', 'recall', 'f1':
                    log['masked'][k]["prediction_eval_metrics"]['seg_bertscore'][typescore] = segm_bertscores[typescore][k]

            #doc-BERTSCORE
            doc_bertscores = self._doc_berstcore(log)
            for k in range(len(log["masked"])):
                log['masked'][k]["prediction_eval_metrics"]['doc_bertscore'] = {}
                for typescore in 'precision', 'recall', 'f1':
                    log['masked'][k]["prediction_eval_metrics"]['doc_bertscore'][typescore] = doc_bertscores[typescore][k]
            """
            log["answ_sim_computed"] = True

    def _calculate_bertscore(self, candidates, references, lang='multilingual'):
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
        return self._calculate_bertscore(predictions, ground_truths)

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
        original_texts += len(hyp_ground_truth_words) * [hyp_text]

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
        original_texts += len(ref_ground_truth_words) * [ref_text]

        assert len(original_texts) == len(new_texts)

        return self._calculate_bertscore(new_texts, original_texts)


    def _calculate_score_from_logs(self, logs):
        scores = []
        if self.use_weighter:
            for log in logs:
                if "hyp" in self.mask_types_to_consider and "ref" in self.mask_types_to_consider:
                    mask_hyp_score = self._weighted_score(log, masking_type="hyp")
                    mask_ref_score = self._weighted_score(log, masking_type="ref")
                    score = np.average([mask_hyp_score, mask_ref_score])
                elif "hyp" in self.mask_types_to_consider:
                    score = self._weighted_score(log, masking_type="hyp")
                elif "ref" in self.mask_types_to_consider:
                    score = self._weighted_score(log, masking_type="ref")
                else:
                    raise ValueError("Specify at least one mask_type to consider when computing scores.")

                scores.append(score)

        elif self.use_sparse_attn:
            for log in logs:
                if "hyp" in self.mask_types_to_consider and "ref" in self.mask_types_to_consider:
                    mask_hyp_score = self._sparse_attn_score(log, masking_type="hyp")
                    mask_ref_score = self._sparse_attn_score(log, masking_type="ref")
                    score = np.average([mask_hyp_score, mask_ref_score])
                elif "hyp" in self.mask_types_to_consider:
                    score = self._sparse_attn_score(log, masking_type="hyp")
                elif "ref" in self.mask_types_to_consider:
                    score = self._sparse_attn_score(log, masking_type="ref")
                else:
                    raise ValueError("Specify at least one mask_type to consider when computing scores.")

                scores.append(score)

        else: #can choose either hyp or ref/src. If both --> average of hyp & ref scores
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
                    raise ValueError("Specify at least one mask_type to consider when computing scores.")

                scores.append(score)

        return scores

    def _sparse_attn_score(self, log, masking_type):
        sorted_score = []
        for mseq in log["masked"]:
            if mseq["masking"] == masking_type:
                word_weight = sum(mseq["weight"]["weight_values"])
                word_score = mseq["prediction_eval_metrics"]["exact_match"]
                word = mseq["ground_truth"]

                sorted_score.append((word_score, word_weight, word))

        sorted_score.sort(key=lambda x: x[1], reverse = True)

        weight_sum = 0
        count = 0
        weighted_score_sum = 0
        for pair in sorted_score:
            weight_sum += pair[1]
            count += 1
            weighted_score_sum += (pair[0] * pair[1])
            if weight_sum > self.attn_threshold:
                break

        self.attn_counts[masking_type].append(count)
        return weighted_score_sum / weight_sum

    def _weighted_score(self, log, masking_type):
        metric_scores = []
        for metric in self.answer_sim_metrics:
            for l in log["masked"]:
                if l["masking"] == masking_type:
                    word_weight = sum(l["weight"]["weight_values"])
                    if metric in l["prediction_eval_metrics"]:
                        metric_scores.append(l["prediction_eval_metrics"][metric] * word_weight)
                    else:
                        logging.warning(
                                f"answer similarity metric {metric} is not in the logs. Setting the metric score to 0. ")
                        metric_scores.append(0)

        return sum(metric_scores)


    def _base_score(self, log, masking_type):
        metric_scores = []
        for metric in self.answer_sim_metrics:
            for l in log["masked"]:
                if l["masking"] == masking_type:
                    if metric in l["prediction_eval_metrics"]:
                        if metric == "seg_bertscore":
                            metric_scores.append(l["prediction_eval_metrics"][metric]["f1"])
                        else:
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
            with open(os.path.join(self.log_dir, hash), 'w', encoding="utf-8") as outfile:
                json.dump(log, outfile, indent=2,  ensure_ascii=False)

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
                       'fill_mask_model_name': "",
                       'weighter_name': "",
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
