from typing import List, Tuple, Dict, Callable
import os
import json
import numpy as np
import logging
from datasets import load_metric
import spacy
import torch
import math
from questeval import DIR, __version__
from questeval.questeval_metric import QuestEval
from questeval.utils import (
    API_T2T,
    sentencize,
    calculate_f1_squad,
    calculate_BERTScore,
    extract_table_answers,
    text2hash
)

HF_ORGANIZATION = "ThomasNLG"
class Mask_QuestEval(QuestEval):
    def __init__(
            self,
            task: str = "text2text",
            language: str = "en",
            answer_types: Tuple = ('NER', 'NOUN'),
            list_scores: Tuple = ('answerability',),
            src_preproc_pipe=None,
            do_weighter: bool = False,
            do_consistency: bool = False,
            qg_batch_size: int = 48,
            clf_batch_size: int = 48,
            limit_sent: int = 5,
            reduction_multi_refs: Callable = max,
            no_cuda: bool = False,
            use_cache: bool = True
    ) -> None:
        super().__init__(
            task,
            language,
            answer_types,
            list_scores,
            src_preproc_pipe,
            do_weighter,
            do_consistency,
            qg_batch_size,
            clf_batch_size,
            limit_sent,
            reduction_multi_refs,
            no_cuda,
            use_cache)
        self.sep = "<sep>"
        self.filter_answ = True
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    def _get_scores(
            self,
            questioned_log: List[Dict],
            compared_log: List[Dict],
            type_score: str
    ) -> List[float]:

        name_model_qg = self._get_qg_hash(compared_log['type'])
        asked_questions = [q for answer_type in self._get_answer_types(compared_log['type'])
                           for q in compared_log['self'][answer_type][name_model_qg]['questions']
                           ]

        name_model_qa = self._get_qa_hash(questioned_log['type'])

        asked_answers = [a for answer_type in self._get_answer_types(compared_log['type'])
                         for a in compared_log['self'][answer_type]['answers']]

        assert len(asked_answers) == len(asked_questions)

        if self.filter_answ:
            qa_pairs = []
            for q, a in list(zip(asked_questions, asked_answers)):
                if a.strip().lower() not in self.stopwords:
                    qa_pairs.append((q, a))
        else:
            qa_pairs = list(zip(asked_questions, asked_answers))

        if type_score == 'answerability':
            scores = [questioned_log['asked'][q][name_model_qa]['answerability']
                      for q, _ in qa_pairs]

        else:  # F1 or BERTScore
            scores = [questioned_log['asked'][q][name_model_qa]['ground_truth'][a][type_score]
                      for q, a in qa_pairs]

        return scores

    def _base_score(
        self,
        questioned_log: Dict,
        compared_log: Dict
    ) -> float:
        regularizer = lambda list_score, list_reg: np.multiply(scores, list_reg).tolist()
        list_borned = lambda a_list: [max(min(1, x), 0) for x in a_list]

        if self.do_consistency:
            consistencies = self._get_scores(compared_log, compared_log, 'f1')

        if self.do_weighter and compared_log['type'] == 'src':
            name_model_qg = self._get_qg_hash(compared_log['type'])
            name_model_weighter = self._get_weighter_hash()
            weighter_probs = [
                w for answer_type in self._get_answer_types(questioned_log['type'])
                for w in compared_log['self'][answer_type][name_model_qg][name_model_weighter]
            ]

        list_scores = []
        for type_score in self.list_scores:
            scores = self._get_scores(questioned_log, compared_log, type_score)

            # if no questions, return a score set to 0; could be improved though ?
            if len(scores) == 0:
                return 0

            # sometimes the answers scores return a value ~1.000000X which is superior to 1
            if type_score == 'bartscore':
                scores = [math.exp(x) for x in scores]
            #    scores = [((x - 0.3169371981312404)/0.2297532889427226) for x in scores]
            #if type_score == 'answerability':
            #    scores = [((x - 0.9638686577722018)/0.13910151937402956) for x in scores]
            #if type_score == 'f1':
            #    scores = [((x - 0.3303939835421951)/0.44225863728196785) for x in scores]

            scores = list_borned(scores)

            if self.do_consistency:
                assert consistencies is not None, "consistencies is None. Please compute the score with ques_consists activate."
                scores = regularizer(scores, consistencies)

            if self.do_weighter and compared_log['type'] == 'src':
                assert weighter_probs is not None, "weighter_probs is None. Please compute the weighter probs with do_weighter activate."
                scores = regularizer(scores, weighter_probs)

            list_scores += scores

        final_score = np.average(list_scores)
        # assert 0 <= final_score <= 1, "score should be in [0-1] "
        return final_score

    def _compute_answer_similarity_scores(
        self,
        logs: Dict,
        type_logs: str
    ) -> None:
        """
        filling the similarity scores
        """

        modified_logs = False
        name_model_qa = self._get_qa_hash(type_logs)
        model_QA = self.models[type_logs]['QA']

        for type_score in self.list_scores:

            # no need for comparison for answerabiliy, it is calculated directly in compute_question_answering
            if type_score == 'answerability':
                continue

            to_do_exs_idxs, to_do_questions, to_do_pred_asws, to_do_gold_asws, to_do_context = [], [], [], [], []
            for idx, log in enumerate(logs):
                if log['text'] == '':
                    continue
                for question in log['asked']:
                    d_answer = log['asked'][question][self._get_qa_hash(log['type'])]
                    for gold_answer in d_answer['ground_truth']:
                        if type_score not in d_answer['ground_truth'][gold_answer]:
                            to_do_exs_idxs += [idx]
                            to_do_questions += [question]
                            to_do_pred_asws += [d_answer['answer']]
                            to_do_gold_asws += [gold_answer]
                            to_do_context += [log['text']]

            if len(to_do_exs_idxs) != 0:

                modified_logs = True

                if type_score == 'f1':
                    sim_scores = [calculate_f1_squad(pred_asw, gold_asw) for pred_asw, gold_asw in
                                  zip(to_do_pred_asws, to_do_gold_asws)]
                elif type_score == 'bertscore':
                    sim_scores = calculate_BERTScore(to_do_pred_asws, to_do_gold_asws, self.metric_BERTScore,
                                                     device=self.device)
                elif type_score == 'bartscore':
                    sources_list = [(question + self.sep + context) for (question, context) in
                               zip(to_do_questions, to_do_context)]
                    sim_scores = model_QA.calculate_bartscore(sources=sources_list, ground_truth_answs=to_do_gold_asws)
                else:
                    raise NotImplementedError(f"{type_score} not implemented")

                assert len(to_do_exs_idxs) == len(sim_scores)
                for i in range(len(to_do_exs_idxs)):
                    idx = to_do_exs_idxs[i]
                    q = to_do_questions[i]
                    a = to_do_gold_asws[i]
                    logs[idx]['asked'][q][name_model_qa]['ground_truth'][a][type_score] = sim_scores[i]

        return modified_logs

    def _load_all_models(self) -> Dict:
        # Textual hypothesis
        models = {"hyp": {}}
        if self.language == 'en':
            models['hyp']['QA'] = f'yliu337/t5_neg_filter_bothcontext'
            models['hyp']['QG'] = f'{HF_ORGANIZATION}/t5-qg_squad1-en'
        else:
            raise("Multilingual evaluation not handled yet.")

        # (if) multimodal sources
        if self.task == "data2text":
            models['src'] = dict()
            models['src']['QA'] = f'{HF_ORGANIZATION}/t5-qa_webnlg_synth-en'
            models['src']['QG'] = f'{HF_ORGANIZATION}/t5-qg_webnlg_synth-en'

        # Loading all the different models
        for modality in models.keys():
            for task in models[modality].keys():
                if not type(models[modality][task]) == str:
                    continue
                models[modality][task]= self.get_model(model_name=models[modality][task])

        # Loading the weighter
        models['Weighter'] = None
        if self.do_weighter:
            models['Weighter'] = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-weighter_cnndm-en')

        # Linking already loaded models for the other keys
        for k in ["src", "ref"]:
            if models.get(k) == None:
                models[k] = dict()
                models[k]['QA'] = models['hyp']['QA']
                models[k]['QG'] = models['hyp']['QG']

        return models

    def _generate_masked_question(self, source: str, chunk: str):
        question = "Placeholder Question"
        sentences = self.spacy_pipeline(source).sents
        for sent in sentences:
            sent_text = sent.text
            if chunk in sent_text:
                question = sent_text.replace(chunk, '<mask>')
                break
        return question

    def _predict_questions(
        self,
        to_do_exs: List[tuple],
        type_logs: str
    ) -> List[str]:

        question_texts = []
        for asw, context in to_do_exs:
            question_texts.append(self._generate_masked_question(source=context, chunk=asw))
        return question_texts

    def get_model(self, model_name: str,):
        keep_score_idx = None

        if True: #'t5' in model_name.lower():
            if 'yliu337' in model_name.lower():
                keep_score_idx = 32102 #t5 special token
            if 'weighter' in model_name.lower():
                # 1176 is the index for the token true in T5 vocabulary
                keep_score_idx = 1176
            if model_name == f"{HF_ORGANIZATION}/t5-qg_squad1-en":
                # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
                self.qg_prefix = 'sv1'

            # batch size
            model_batch_size = self.qg_batch_size if "qg" in model_name.lower() else self.clf_batch_size

            print(model_name.lower())            
            print(keep_score_idx)
            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                keep_score_idx=keep_score_idx,
                max_source_length=512,
                model_batch_size=model_batch_size,
                device=self.device
            )

        else:
            raise NotImplementedError(f'Model Name Not Handled: the model name should contain t5 ({model_name}).')

        return model

class Mask_QuestEval_src(Mask_QuestEval):
    def __init__(
            self,
            task: str = "text2text",
            language: str = "en",
            answer_types: Tuple = ('NER', 'NOUN'),
            list_scores: Tuple = ('answerability',),
            src_preproc_pipe=None,
            do_weighter: bool = False,
            do_consistency: bool = False,
            qg_batch_size: int = 48,
            clf_batch_size: int = 48,
            limit_sent: int = 5,
            reduction_multi_refs: Callable = max,
            no_cuda: bool = False,
            use_cache: bool = True
    ) -> None:
        super().__init__(
            task,
            language,
            answer_types,
            list_scores,
            src_preproc_pipe,
            do_weighter,
            do_consistency,
            qg_batch_size,
            clf_batch_size,
            limit_sent,
            reduction_multi_refs,
            no_cuda,
            use_cache)
        self.sep = "<sep>"

    def _calculate_score_from_logs(
            self,
            hyp_log: List[Dict],
            compared_logs: List[List[Dict]]
    ) -> float:

        scores = []
        for compared_log in compared_logs:
            if compared_log['text'] == '' or hyp_log['text'] == '':
                score = 0
            else:
                compared_score = self._base_score(compared_log, hyp_log)
                score = compared_score
            scores.append(score)
        return self.reduction_multi_refs(scores)

class Mask_QuestEval_hyp(Mask_QuestEval):
    def __init__(
            self,
            task: str = "text2text",
            language: str = "en",
            answer_types: Tuple = ('NER', 'NOUN'),
            list_scores: Tuple = ('answerability',),
            src_preproc_pipe=None,
            do_weighter: bool = False,
            do_consistency: bool = False,
            qg_batch_size: int = 48,
            clf_batch_size: int = 48,
            limit_sent: int = 5,
            reduction_multi_refs: Callable = max,
            no_cuda: bool = False,
            use_cache: bool = True
    ) -> None:
        super().__init__(
            task,
            language,
            answer_types,
            list_scores,
            src_preproc_pipe,
            do_weighter,
            do_consistency,
            qg_batch_size,
            clf_batch_size,
            limit_sent,
            reduction_multi_refs,
            no_cuda,
            use_cache)
        self.sep = "<sep>"

    def _calculate_score_from_logs(
        self,
        hyp_log: List[Dict],
        compared_logs: List[List[Dict]]
    ) -> float:

        scores = []
        for compared_log in compared_logs:
            if compared_log['text'] == '' or hyp_log['text'] == '':
                score = 0
            else:
                hyp_score = self._base_score(hyp_log, compared_log)
                score = hyp_score
            scores.append(score)
        return self.reduction_multi_refs(scores)
