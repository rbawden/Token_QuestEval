from typing import Dict, List, Optional, Tuple
import string
import re
import unidecode
import collections
import torch
import torch.nn as nn
import hashlib
from transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.models.mt5 import MT5ForConditionalGeneration, MT5TokenizerFast
import spacy
from spacy.lang.id import Indonesian
from spacy.lang.tr import Turkish
import os

MODEL_CLASSES = {
    "t5": (T5ForConditionalGeneration, T5TokenizerFast),
    "mt5": (MT5ForConditionalGeneration, MT5TokenizerFast),
}

SPACY_PIPELINE_NAMES = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "ru": "ru_core_news_sm",
    "zh": "zh_core_web_sm"
}

SPACY_PIPELINES = {}

for lang in SPACY_PIPELINE_NAMES:
    pipeline_name = SPACY_PIPELINE_NAMES[lang]
    try:
        pipeline = spacy.load(pipeline_name)
    except OSError:
        from spacy.cli import download
        download(pipeline_name)
        pipeline = spacy.load(pipeline_name)
    SPACY_PIPELINES[lang] = pipeline

SPACY_PIPELINES["id"] = Indonesian()
SPACY_PIPELINES["tr"] = Turkish()


def resize(original_list, desired_len, stop_index):
    if len(original_list) > desired_len:
        output = original_list[:(desired_len - 1)] + [stop_index]
    elif len(original_list) < desired_len:
        output = original_list[:desired_len] + [0] * (desired_len - len(original_list))
    else:
        output = original_list

    return output


def add_label_padding(label_list):
    lengths = [len(l) for l in label_list]
    max_len = max(lengths)
    padded_label_list = []

    for l in label_list:
        if len(l) < max_len:
            padded_label_list.append(l + (max_len - len(l)) * [0])
        else:
            padded_label_list.append(l)
    return padded_label_list

class API_T2T:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_seq_length: int,
        model_batch_size: int,
        sliding_window_size: int,
        order: str = "ref_first",
        device: str = "cuda"
    ) -> None:

        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if 'mt5' in pretrained_model_name_or_path:
            model_type = "mt5"
        else:
            model_type = 't5'

        model_class, model_tokenizer = MODEL_CLASSES[model_type]

        self.tokenizer = model_tokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model = model_class.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        self.sliding_window_size = sliding_window_size
        self.max_seq_length = max_seq_length
        self.model_batch_size = model_batch_size
        self.text_pair_batch_size = 256

        if device == "cuda":
            self.model.cuda()

        self.order = order
        self.tokenizer_indices = {
            "<extra_id_0>": self.tokenizer.convert_tokens_to_ids("<extra_id_0>"),
            "<extra_id_1>": self.tokenizer.convert_tokens_to_ids("<extra_id_1>"),
            "<sep>": self.tokenizer.convert_tokens_to_ids("<sep>"),
            "</s>": self.tokenizer.convert_tokens_to_ids("</s>")
        }

        #self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        #self.lsm = nn.LogSoftmax(dim=1)

    def get_word_list(self, sentence, spacy_tokenizer):
        t5_tokenizer = self.tokenizer

        t5_result = t5_tokenizer(sentence, return_offsets_mapping=True)
        t5_tok = t5_result['input_ids']
        spacy_tok = [(t.text, t.idx, t.pos_) for t in spacy_tokenizer(sentence)]

        # Get the mapping for both t5 and spacy tokenization, and then the INTERSECTION of both
        t5_idxs = [t[1] for t in t5_result['offset_mapping']]

        t5_tok_idx_pairs = list(zip(t5_tok, t5_idxs))
        spacy_idxs = [t[1] + len(t[0]) for t in spacy_tok]

        # Get the mapping between POS_tags and new spacy idx
        pos_idx_mapping = []
        for t, idx in zip(spacy_tok, spacy_idxs):
            pos_idx_mapping.append((t[2], idx))

        # Compute intersection
        intersection = sorted(list(set(t5_idxs) & set(spacy_idxs)))

        # Create dict of words
        word_dict = {}
        sublist = []

        for tok, idx in t5_tok_idx_pairs:
            sublist.append(tok)
            if idx in intersection:
                if idx not in word_dict:
                    word_dict[idx] = sublist
                else:
                    word_dict[idx] = word_dict[idx] + sublist
                sublist = []

        # Create list of POS tags
        pos_dict = {}
        sublist = []
        for pos_tag, idx in pos_idx_mapping:
            sublist.append(pos_tag)
            if idx in intersection:
                if idx not in pos_dict:
                    pos_dict[idx] = sublist
                else:
                    pos_dict[idx] = pos_dict[idx] + sublist
                sublist = []

        return list(word_dict.values()), list(pos_dict.values())

    def get_all_masked_sequences(self, text_pair):
        hypothesis = text_pair["hypothesis"]
        reference = text_pair["reference"]
        hyp_lang = text_pair["hyp_lang"]
        ref_lang = text_pair["ref_lang"]

        # get the list of words (intersection of spaCy and T5 tokenization)
        hyp_word_list, hyp_pos_tags = self.get_word_list(hypothesis, SPACY_PIPELINES[hyp_lang])
        ref_word_list, ref_pos_tags = self.get_word_list(reference, SPACY_PIPELINES[ref_lang])

        ref_tokens = [item for sublist in ref_word_list for item in sublist]
        hyp_tokens = [item for sublist in hyp_word_list for item in sublist]

        all_masked_seqs = []

        # loop over all words in hypothesis
        for index in range(len(hyp_word_list)):
            label = [self.tokenizer_indices["<extra_id_0>"]] + hyp_word_list[index] \
                    + [self.tokenizer_indices["<extra_id_1>"], self.tokenizer_indices["</s>"]]
            label_pos = hyp_pos_tags[index]

            copy_hyp_word_list = hyp_word_list.copy()
            copy_hyp_word_list[index] = self.tokenizer_indices["<extra_id_0>"]

            # get list of tokens before and after the masked word
            word_list_before = copy_hyp_word_list[0:index]
            token_list_before = [item for sublist in word_list_before for item in sublist]
            word_list_after = copy_hyp_word_list[index + 1:]
            token_list_after = [item for sublist in word_list_after for item in sublist]

            # apply sliding window
            token_list_before = token_list_before[-self.sliding_window_size:]
            token_list_after = token_list_after[:self.sliding_window_size]
            masked_seq = token_list_before + [self.tokenizer_indices["<extra_id_0>"]] + token_list_after

            # compute nb of tokens left for context
            ref_length = self.max_seq_length - len(masked_seq) - 2
            ref_tokens = ref_tokens[:ref_length]

            mask_dict = {"masking": "hyp",
                         "label_pos": label_pos,
                         "label_tokens": label,
                         "hyp_tokens": masked_seq,
                         "ref_tokens": ref_tokens}

            all_masked_seqs.append(mask_dict)

        # loop over all words in reference
        for index in range(len(ref_word_list)):
            label = [self.tokenizer_indices["<extra_id_0>"]] + ref_word_list[index] \
                    + [self.tokenizer_indices["<extra_id_1>"], self.tokenizer_indices["</s>"]]
            label_pos = ref_pos_tags[index]

            copy_ref_word_list = ref_word_list.copy()
            copy_ref_word_list[index] = self.tokenizer_indices["<extra_id_0>"]

            # get list of tokens before and after the masked word
            word_list_before = copy_ref_word_list[0:index]
            token_list_before = [item for sublist in word_list_before for item in sublist]
            word_list_after = copy_ref_word_list[index + 1:]
            token_list_after = [item for sublist in word_list_after for item in sublist]

            # apply sliding window
            token_list_before = token_list_before[-self.sliding_window_size:]
            token_list_after = token_list_after[:self.sliding_window_size]
            masked_seq = token_list_before + [self.tokenizer_indices["<extra_id_0>"]] + token_list_after

            # compute nb of tokens left for context
            hyp_length = self.max_seq_length - len(masked_seq) - 2
            hyp_tokens = hyp_tokens[:hyp_length]

            mask_dict = {"masking": "ref",
                         "label_pos": label_pos,
                         "label_tokens": label,
                         "hyp_tokens": hyp_tokens,
                         "ref_tokens": masked_seq}

            all_masked_seqs.append(mask_dict)
        return all_masked_seqs

    def preprocess_batch(self, batch):
        # set order = "focus-first" to do: focus <sep> context
        # set order = "hyp_first" to do: hypothesis <sep> reference/source
        order = self.order
        input_batch = {
            "input_ids": [],
            "attention_mask": [],
            "label_ids": []
        }
        str_labels = []
        mask_tags = []
        pos_tags = []
        word_numbers = []

        for text_pair in batch:
            all_masked_seqs = self.get_all_masked_sequences(text_pair)
            word_numbers.append(len(all_masked_seqs))
            for mask_dict in all_masked_seqs:
                if order == "hyp_first":
                    input_ids = mask_dict["hyp_tokens"] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                        "ref_tokens"] + [self.tokenizer_indices["</s>"]]
                elif order == "ref_first":
                    input_ids = mask_dict["ref_tokens"][:-1] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                        "hyp_tokens"][:-1] + [self.tokenizer_indices["</s>"]]
                else:
                    if mask_dict["masking"] == "hyp":
                        input_ids = mask_dict["hyp_tokens"] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                            "ref_tokens"] + [self.tokenizer_indices["</s>"]]
                    else:  # mask_dict["masking"] == "ref"
                        input_ids = mask_dict["ref_tokens"] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                            "hyp_tokens"] + [self.tokenizer_indices["</s>"]]

                attention_mask = len(input_ids) * [1]
                input_ids = resize(input_ids, self.max_seq_length, self.tokenizer_indices["</s>"])
                attention_mask = resize(attention_mask, self.max_seq_length, self.tokenizer_indices["</s>"])
                str_label = self.tokenizer.decode(mask_dict["label_tokens"], skip_special_tokens=True)

                input_batch["input_ids"].append(input_ids)
                input_batch["attention_mask"].append(attention_mask)
                input_batch["label_ids"].append(mask_dict["label_tokens"])

                str_labels.append(str_label)
                pos_tags.append(mask_dict["label_pos"])
                mask_tags.append(mask_dict["masking"])


        no_padded_label_ids = input_batch["label_ids"]
        input_batch["label_ids"] = add_label_padding(input_batch["label_ids"])

        return input_batch, no_padded_label_ids, str_labels, mask_tags, word_numbers, pos_tags
    
    def predict(self, text_pairs: dict):
        # text_pairs should be a list of dict, each dict being of the form:
        #{"hypothesis": str,
        # "reference": str,
        # "hyp_lang": str,
        # "ref_lang": str}

        preds = []
        all_labels = []
        all_mask_tags = []
        all_pos_tags = []
        all_word_numbers = [] #number each text pairs produced
        all_pred_scores = []
        all_gold_scores = []
        all_gold_label_tok_ids = []
        all_gold_label_toks = []
        all_pred_tok_ids = []
        all_pred_toks = []
        all_label_ids = []

        for i in range(0, len(text_pairs), self.text_pair_batch_size):

            batch = text_pairs[i: i+self.text_pair_batch_size]
            input_batch, no_padded_label_ids, str_labels, mask_tags, word_numbers, pos_tags = self.preprocess_batch(batch)
            all_label_ids = all_label_ids + no_padded_label_ids
            all_word_numbers = all_word_numbers + word_numbers
            all_labels = all_labels + str_labels
            all_mask_tags = all_mask_tags + mask_tags
            all_pos_tags = all_pos_tags + pos_tags

            os.sys.stderr.write(str(i) + ', lentextpairs=' + str(len(text_pairs)) + ', textpairbsz=' + str(self.text_pair_batch_size) + '\n')
            for k in range(0, len(str_labels), self.model_batch_size):
                os.sys.stderr.write('\tk=' + str(k) + ', lenlabels=' + str(len(str_labels)) + ', modelbsz=' + str(self.model_batch_size) + '\n')

                with torch.no_grad():
                    input_ids = torch.tensor(input_batch["input_ids"][k: k+self.model_batch_size])
                    attention_mask = torch.tensor(input_batch["attention_mask"][k: k+self.model_batch_size])
                    dict_generated_ids = self.model.generate(
                        input_ids=input_ids.to(self.model.device),
                        attention_mask=attention_mask.to(self.model.device),
                        use_cache=True,
                        decoder_start_token_id=None,
                        num_beams=1,
                        num_return_sequences=1,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True
                    )

                    
                    these_pred_scores = self.extract_pred_label_scores(dict_generated_ids['scores'], dict_generated_ids['sequences'])
                    all_pred_scores.extend(these_pred_scores)

                    #import pdb; pdb.set_trace()
                    # generate answers + compute answerability
                    gen_text = self.tokenizer.batch_decode(
                        dict_generated_ids['sequences'],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    preds += gen_text

                    # get scores of gold labels
                    gold_label_ids = torch.tensor(input_batch["label_ids"][k: k+self.model_batch_size])
                    gold_logits = self.model(input_ids=input_ids.to(self.model.device),
                                             labels=gold_label_ids.to(self.model.device),
                                             attention_mask=attention_mask.to(self.model.device)).logits
                    list_gold_logits = [gold_logits[x] for x in range(gold_logits.shape[0])]
                    these_gold_scores = self.extract_gold_label_scores(list_gold_logits, gold_label_ids.to(self.model.device))
                    all_gold_scores.extend(these_gold_scores)


                    #import pdb; pdb.set_trace()
                    # subword tokens
                    ignore = [32099, 32098, 0, 1]
                    for example in gold_label_ids:
                        gold_label_tok_ids = [x.item() for x in example if x not in ignore]
                        all_gold_label_tok_ids.append(gold_label_tok_ids)
                        all_gold_label_toks.append(self.tokenizer.convert_ids_to_tokens(gold_label_tok_ids))
                    for example in dict_generated_ids['sequences']:
                        pred_tok_ids = [x.item() for x in example if x not in ignore]
                        all_pred_tok_ids.append(pred_tok_ids)
                        all_pred_toks.append(self.tokenizer.convert_ids_to_tokens(pred_tok_ids))
                        
                    import pdb; pdb.set_trace()

        assert len(preds) == len(all_labels)
        assert len(preds) == len(all_mask_tags)
        assert len(preds) == len(all_pos_tags)
        assert len(all_word_numbers) == len(text_pairs)
        assert sum(all_word_numbers) == len(preds)
        assert len(all_pred_scores) == len(all_labels)
        assert len(all_gold_scores) == len(all_labels)
        assert len(all_gold_label_toks) == len(all_labels)
        assert len(all_gold_label_tok_ids) == len(all_labels)
        assert len(all_pred_toks) == len(all_labels)
        assert len(all_pred_tok_ids) == len(all_labels)

        outputs = []

        for k in range(len(preds)):
            output = {
                "prediction": preds[k],
                "ground_truth": all_labels[k],
                "ground_truth_ids": all_gold_label_tok_ids[k],
                "ground_truth_toks": all_gold_label_toks[k],
                "prediction_ids": all_pred_tok_ids[k],
                "prediction_toks": all_pred_toks[k],
                "ground_truth_ids": all_label_ids[k][1:-2], #to not keep extra + eos tokens
                "masking": all_mask_tags[k],
                "prediction_eval_metrics": {
                    "pred_scores": all_pred_scores[k],
                    "gold_scores": all_gold_scores[k]
                },
                "POS_tag": all_pos_tags[k], 
                "weight": 0.0
            }

            outputs.append(output)

        categorized_outputs = []
        left_index = 0
        right_index = 0
        indices = []

        for i in range(len(all_word_numbers)):
            right_index = right_index + all_word_numbers[i]
            indices.append((left_index, right_index))
            left_index = right_index

        for (l, r) in indices:
            all_outputs = outputs[l:r]
            filtered_outputs = []
            for o in all_outputs:
                if len(o["ground_truth"]) > 0: #filter out ground truths of empty strings
                    filtered_outputs.append(o)
            categorized_outputs.append(filtered_outputs)

        return categorized_outputs

    def extract_gold_label_scores(self, all_scores, all_idxs):
        '''
        all_scores: list of n tensors (k,v)
                        where n=num examples, k=number of toks, v=vocab size
        all_idxs: (n, k) tensor 
        '''
        all_label_scores = []
        # go through all examples (one list per example)
        for ex_id in range(len(all_scores)):
            # get softmax scores for each token in example
            example_scores = nn.functional.log_softmax(all_scores[ex_id], dim=-1)
            # get gold label indices for each token in example
            idxs = all_idxs[ex_id]
            # select the scores corresponding to the indices of the predicted subwords
            label_scores = example_scores.gather(-1, idxs.unsqueeze(-1)).squeeze(-1)
            # store scores for each token (ignoring padding and special tokens)
            all_label_scores.append([label_scores[s].item() for s in range(len(label_scores)) if idxs[s] not in [0, 1, 32099, 32098]])

        return all_label_scores

    def extract_pred_label_scores(self, all_scores, all_idxs):
        '''
        all_scores: list of k tensors (n, v)
                        where n=num examples, k=number of toks, v=vocab size
        all_idxs: (n, k) tensor
        '''
        # prediction scores (start from index 2 each time)
        all_label_scores = [[] for i in range(len(all_idxs))]
        # go through from 0 to max length of predictions 
        # (ignore first 2, which are always the same). 
        # Also, the scores do not contain scores for the first index, so use i-1 for indexing
        for tok_id in range(2, len(all_scores)):
            # softmax scores first
            example_scores = nn.functional.log_softmax(all_scores[tok_id-1], dim=-1)
            idxs = all_idxs.t()[tok_id]
            # select the scores corresponding to the indices of the predicted subwords
            label_scores = example_scores.gather(-1, idxs.unsqueeze(-1)).squeeze(-1)
            # store prediction scores for each predicted subword
            for p in range(label_scores.shape[-1]):
                # exclude special tokens and padding
                if idxs[p].item() not in [0, 1, 32099, 32098]:
                    all_label_scores[p].append(label_scores[p].item())
        return all_label_scores
