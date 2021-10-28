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
class Token_QuestEval(QuestEval):
    def __init__(
            self,
            task: str = "text2text",
            language: str = "en",
            answer_types: Tuple = ('TOKEN',),
            list_scores: Tuple = ('f1', 'bartscore',),
            doc_types: Tuple = ('mask_src', 'mask_hyp'),
            sliding_window_size: int = 24,
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
        self.doc_types = doc_types
        self.sliding_window_size = sliding_window_size
        self.bartscore = 'bartscore' in list_scores

        #Below are obsolete attributes to be deleted along with some parts of the codes
        self.filter_answ = False
        self.filter_pos = False
        self.wanted_pos = ["VERB", "NOUN", "PROPN"]
        self.stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    ## "MAIN" METHOD: apply MaskEval/QuestEval to a batch, made of list of hyp & list of src #####################
    def _batch_questeval(
            self,
            hypothesis: List[str],
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> List[float]:

        list_compared_logs = []
        d_loaded_logs = dict()

        # Hypothesis
        hyp_logs, hyp_hashes, modified_logs = self._texts2logs(hypothesis, type_logs='hyp', d_loaded_logs=d_loaded_logs)
        if modified_logs:
            self._serialize_logs(hyp_logs, hyp_hashes)

        # Source
        src_logs, src_hashes, modified_logs = self._texts2logs(sources, type_logs='src', d_loaded_logs=d_loaded_logs)
        # Asking the questions on the compared text
        modified_logs = max(self._compute_question_answering(src_logs, hyp_logs, 'src', 'hyp'), modified_logs)
        modified_logs = max(self._compute_question_answering(hyp_logs, src_logs, 'hyp', 'src'), modified_logs)
        # Compute the similarity scores
        modified_logs = max(self._compute_answer_similarity_scores(src_logs, type_logs='src'), modified_logs)
        # Serialise logs
        if modified_logs:
            self._serialize_logs(src_logs, src_hashes)
            self._serialize_logs(hyp_logs, hyp_hashes)
        list_compared_logs.append(src_logs)

        ### reference removed for now ###

        # Compute the similarity scores for hyp
        modified_logs = self._compute_answer_similarity_scores(hyp_logs, type_logs='hyp')
        # Serialise hyp logs
        if modified_logs:
            self._serialize_logs(hyp_logs, hyp_hashes)

        list_compared_logs = [
            [
                list_compared_logs[i][j]
                for i in range(len(list_compared_logs))
            ]
            for j in range(len(list_compared_logs[0]))
        ]

        # Calculate Score
        scores = []
        for hyps_log, compared_logs in zip(hyp_logs, list_compared_logs):
            scores.append(self._calculate_score_from_logs(hyps_log, compared_logs))

        return scores

    ## Load and define finetuned language model to be used ########################################################
    def _load_all_models(self) -> Dict:
        # Textual hypothesis
        models = {"hyp": {}}
        if self.language == 'en':
            models['hyp']['QA'] = f'yliu337/sliding_window_token_both_ctx'
            models['hyp']['QG'] = f'{HF_ORGANIZATION}/t5-qg_squad1-en'
        else:
            raise ("Multilingual evaluation not handled yet.")

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
                models[modality][task] = self.get_model(model_name=models[modality][task])

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

    def get_model(self, model_name: str, ):
        keep_score_idx = None

        if True:  # 't5' in model_name.lower():
            if 'yliu337' in model_name.lower():
                keep_score_idx = 32102  # t5 special token
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

    ## Preprocessing: write into logs masked segments & ground truth labels #######################################
    def _texts2logs(
        self,
        texts: List[str],
        type_logs: str,
        d_loaded_logs: Dict
    ):
        modified_logs = False

        # Preprocessing
        if type_logs == 'src' and self.src_preproc_pipe is not None:
            texts = [self.src_preproc_pipe(source) for source in texts]

        logs, logs_hashes = self._load_logs(texts, type_logs, d_loaded_logs)
        # Selecting the question answer pairs
        modified_logs = max(self._get_question_answers(logs, type_logs), modified_logs)
        # Asking the questions on itself (Round trip consistency)
        if self.do_consistency:
            modified_logs = (self._compute_question_answering(logs, logs, type_logs, type_logs), modified_logs)
        # Weighter
        if type_logs == 'src' and self.do_weighter:
            modified_logs = max(self._compute_weighter(logs, type_logs='src'), modified_logs)

        return logs, logs_hashes, modified_logs

    def _load_logs(
        self,
        texts: List,
        type_logs: str,
        d_loaded_logs: Dict
    ) -> Tuple[List[Dict], List[str]]:
        logs, log_hashs = [], []

        for text in texts:
            #text = ' '.join(text.split()[:self.text_len_limit])
            log_hash = text2hash(text)
            if log_hash not in d_loaded_logs:
                log = {'type': type_logs, 'text': text, 'self': dict(), 'asked': dict()}
                if not (self.use_cache and log_hash in self.hash_files and text != ""):
                    temp=1
                if self.use_cache and log_hash in self.hash_files and text != "":
                    cached_path = os.path.join(self.log_dir, log_hash)
                    try:
                        with open(cached_path, 'r') as f_log:
                            tmp  = json.load(f_log)
                            assert all([k in log for k in ['type', 'text', 'self', 'asked']])
                            assert isinstance(log['type'], str)
                            assert isinstance(log['text'], str)
                            assert isinstance(log['self'], dict)
                            assert isinstance(log['asked'], dict)
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

    def _get_qas(self, doc):
        PADDING_SIZE = self.sliding_window_size
        spacy_doc = self.spacy_pipeline(doc)
        tokens = [token.text for token in spacy_doc]
        pos_tags = [token.pos_ for token in spacy_doc]

        list_questions = []
        list_answers = []

        for i in range(len(tokens)):
            asw_token = tokens[i]
            pos_tag = pos_tags[i]
            masked_text = ' '.join(tokens[max(0, i - PADDING_SIZE):i]) + ' <mask> ' + ' '.join(tokens[i + 1: i + PADDING_SIZE + 1])
            list_answers.append({'text': asw_token, 'pos_tag': pos_tag})
            list_questions.append(masked_text)

        return list_questions, list_answers

    def _get_qa_pairs(
            self,
            to_do_exs: List[tuple],
    ) -> List[str]:

        answers = []
        question_texts = []

        for text in to_do_exs:
            doc = self.spacy_pipeline(text)

            trimmed_doc = ' '.join([sent.text for sent in doc.sents][:self.limit_sent])
            doc_q, doc_a = self._get_qas(trimmed_doc)

            answers.append(doc_a)
            question_texts.append(doc_q)

        return answers, question_texts

    def _get_question_answers(
            self,
            logs: List[Dict],
            type_logs: str
    ) -> None:
        name_model_qg = self._get_qg_hash(type_logs)

        to_do_exs, to_do_exs_idxs, to_do_exs_types = [], [], []
        for idx, log in enumerate(logs):
            if log['text'] == '':
                continue
            for answer_type in self._get_answer_types(type_logs):
                if answer_type not in log['self'] and log['text'] != '':
                    log['self'][answer_type] = dict()

                if name_model_qg not in log['self'][answer_type]:
                    log['self'][answer_type][name_model_qg] = {'questions': []}

                    to_do_exs += [log['text']]
                    to_do_exs_idxs += [idx]
                    to_do_exs_types += [answer_type]

        if len(to_do_exs) != 0:
            answers, question_texts = self._get_qa_pairs(to_do_exs)
            for i in range(len(question_texts)):
                idx = to_do_exs_idxs[i]
                answer_type = to_do_exs_types[i]

                logs[idx]['self'][answer_type][name_model_qg]['questions'] = question_texts[i]
                logs[idx]['self'][answer_type]['answers'] = answers[i]

        return len(to_do_exs) != 0

    ## Run prediction to get the model pred, bartscore, etc. #######################################################
    def _predict_answers(
            self,
            to_do_exs: List[tuple],
            type_logs: str
    ) -> Tuple[List[float], List[str]]:
        model_QA = self.models[type_logs]['QA']
        formated_inputs = [f'{question} {self.sep} {context}' for question, context, _ in to_do_exs]
        labels = [f'{gold_answer}' for _, _, gold_answer in to_do_exs]

        bartscores, ans_scores, qa_texts = model_QA.predict(formated_inputs, labels, self.bartscore)

        return bartscores, ans_scores, qa_texts

    def _compute_question_answering(
            self,
            logs_1: Dict,
            logs_2: Dict,
            type_logs_1: str,
            type_logs_2: str
    ) -> None:
        """
        asking questions from logs_2 on text from logs_1
        """
        assert len(logs_1) == len(logs_2)

        name_model_qg = self._get_qg_hash(type_logs_2)
        name_model_qa = self._get_qa_hash(type_logs_1)

        to_do_exs, to_do_exs_types, to_do_exs_idxs, to_do_gold_asws = [], [], [], []
        for idx, (log_1, log_2) in enumerate(zip(logs_1, logs_2)):
            if log_1['text'] == '' or log_2['text'] == '':
                continue
            for answer_type in self._get_answer_types(type_logs_2):
                questions = log_2['self'][answer_type][name_model_qg]['questions']
                gold_answers = log_2['self'][answer_type]['answers']

                assert len(questions) == len(gold_answers)

                for question, gold_answer in zip(questions, gold_answers):
                    if question not in log_1['asked']:
                        log_1['asked'][question] = dict()

                    if name_model_qa not in log_1['asked'][question]:
                        to_do_exs += [(question, log_1['text'], gold_answer['text'])]
                        to_do_exs_idxs += [idx]
                        to_do_gold_asws += [gold_answer]

                    # if already in the logs, we need to add the gold_answers if it hasnt been yet
                    elif gold_answer['text'] not in log_1['asked'][question][name_model_qa]['ground_truth']:
                        log_1['asked'][question][name_model_qa]['ground_truth'][gold_answer['text']] = {}

        if len(to_do_exs) != 0:
            # to modify
            bartscores, answerability_scores, qa_texts = self._predict_answers(to_do_exs, type_logs_1)

            assert len(to_do_exs) == len(qa_texts) == len(to_do_gold_asws) == len(answerability_scores) == len(
                bartscores)
            for i in range(len(to_do_exs)):

                question = to_do_exs[i][0]
                idx = to_do_exs_idxs[i]
                assert to_do_exs[i][1] == logs_1[idx]['text']

                if name_model_qa not in logs_1[idx]['asked'][question]:
                    logs_1[idx]['asked'][question][name_model_qa] = {'answer': qa_texts[i],
                                                                     'answerability': answerability_scores[i],
                                                                     'bartscore': math.exp(bartscores[i]),
                                                                     'ground_truth': dict()
                                                                     }
                logs_1[idx]['asked'][question][name_model_qa]['ground_truth'][to_do_gold_asws[i]['text']] = {
                    'pos_tag': to_do_gold_asws[i]['pos_tag']}

        return len(to_do_exs) != 0

    ## Compute scores from logs ##############################################################################
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
                if 'mask_src' in self.doc_types:
                    hyp_score = self._base_score(hyp_log, compared_log)
                    score = hyp_score
                if 'mask_hyp' in self.doc_types:
                    compared_score = self._base_score(compared_log, hyp_log)
                    score = compared_score
                if 'mask_hyp' in self.doc_types and 'mask_src' in self.doc_types:
                    score = np.average([hyp_score, compared_score])
            scores.append(score)
        return self.reduction_multi_refs(scores)

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

        asked_answers = [a['text'] for answer_type in self._get_answer_types(compared_log['type'])
                         for a in compared_log['self'][answer_type]['answers']]

        asked_answers_with_pos = [a for answer_type in self._get_answer_types(compared_log['type'])
                         for a in compared_log['self'][answer_type]['answers']]

        assert len(asked_answers) == len(asked_questions)

        if self.filter_answ:
            qa_pairs = []
            for q, a in list(zip(asked_questions, asked_answers)):
                if a.strip().lower() not in self.stopwords:
                    qa_pairs.append((q, a))
        elif self.filter_pos:
            qa_pairs = []
            for q, a in list(zip(asked_questions, asked_answers_with_pos)):
                if a['pos_tag'] in self.wanted_pos:
                    qa_pairs.append((q, a['text']))
        else:
            qa_pairs = list(zip(asked_questions, asked_answers))

        if type_score in ['answerability', 'bartscore']:
            scores = [questioned_log['asked'][q][name_model_qa][type_score]
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
            scores = list_borned(scores)

            if self.do_consistency:
                assert consistencies is not None, "consistencies is None. Please compute the score with ques_consists activate."
                scores = regularizer(scores, consistencies)

            if self.do_weighter and compared_log['type'] == 'src':
                assert weighter_probs is not None, "weighter_probs is None. Please compute the weighter probs with do_weighter activate."
                scores = regularizer(scores, weighter_probs)

            list_scores += scores

        final_score = np.average(list_scores)
        assert 0 <= final_score <= 1, "score should be in [0-1] "
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

            # no need for comparison for answerabiliy and bartscore, it is calculated directly in compute_question_answering
            if type_score in ['answerability', 'bartscore']:
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
                else:
                    raise NotImplementedError(f"{type_score} not implemented")

                assert len(to_do_exs_idxs) == len(sim_scores)
                for i in range(len(to_do_exs_idxs)):
                    idx = to_do_exs_idxs[i]
                    q = to_do_questions[i]
                    a = to_do_gold_asws[i]
                    logs[idx]['asked'][q][name_model_qa]['ground_truth'][a][type_score] = sim_scores[i]

        return modified_logs


