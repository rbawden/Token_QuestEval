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

class API_T2T:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_seq_length: int,
        model_batch_size: int,
        sliding_window_size: int,  # Note: will work only if beamsize == 1
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

        if device == "cuda":
            self.model.cuda()

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
        spacy_tok = [(t.text, t.idx) for t in spacy_tokenizer(sentence)]

        # Get the mapping for both t5 and spacy tokenization, and then the INTERSECTION of both
        t5_idxs = [t[1] for t in t5_result['offset_mapping']]

        t5_tok_idx_pairs = list(zip(t5_tok, t5_idxs))
        spacy_idxs = [t[1] + len(t[0]) for t in spacy_tok]

        intersection = sorted(list(set(t5_idxs) & set(spacy_idxs)))

        word_list = []
        sublist = []
        for tok, idx in t5_tok_idx_pairs:
            sublist.append(tok)
            if idx in intersection:
                word_list.append(sublist)
                sublist = []

        return word_list

    def get_all_masked_sequences(self, text_pair):
        hypothesis = text_pair["hypothesis"]
        reference = text_pair["reference"]
        hyp_lang = text_pair["hyp_lang"]
        ref_lang = text_pair["ref_lang"]
        t5_tokenizer = self.tokenizer

        # get the list of words (intersection of spaCy and T5 tokenization)
        hyp_word_list = self.get_word_list(hypothesis, SPACY_PIPELINES[hyp_lang])
        ref_word_list = self.get_word_list(reference, SPACY_PIPELINES[ref_lang])

        ref_tokens = [item for sublist in ref_word_list for item in sublist]
        hyp_tokens = [item for sublist in hyp_word_list for item in sublist]

        all_masked_seqs = []

        # loop over all words in hypothesis
        for index in range(len(hyp_word_list)):
            label = [self.tokenizer_indices["<extra_id_0>"]] + hyp_word_list[index] \
                    + [self.tokenizer_indices["<extra_id_1>"], self.tokenizer_indices["</s>"]]

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
                         "label_tokens": label,
                         "hyp_tokens": masked_seq,
                         "ref_tokens": ref_tokens}

            all_masked_seqs.append(mask_dict)

        # loop over all words in reference
        for index in range(len(ref_word_list)):
            label = [self.tokenizer_indices["<extra_id_0>"]] + ref_word_list[index] \
                    + [self.tokenizer_indices["<extra_id_1>"], self.tokenizer_indices["</s>"]]

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
                         "label_tokens": label,
                         "hyp_tokens": hyp_tokens,
                         "ref_tokens": masked_seq}

            all_masked_seqs.append(mask_dict)

        return all_masked_seqs

    def preprocess_batch(self, batch, order="hyp_first"):
        # set order = "focus-first" to do: focus <sep> context
        # set order = "hyp_first" to do: hypothesis <sep> reference/source

        input_batch = {
            "input_ids": [],
            "attention_mask": []
        }
        labels = []
        mask_tags = []
        word_numbers = []

        for text_pair in batch:
            all_masked_seqs = self.get_all_masked_sequences(text_pair)
            word_numbers.append(len(all_masked_seqs))
            for mask_dict in all_masked_seqs:
                if order == "hyp_first":
                    input_ids = mask_dict["hyp_tokens"] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                        "ref_tokens"] + [self.tokenizer_indices["</s>"]]
                else:
                    if mask_dict["masking"] == "hyp":
                        input_ids = mask_dict["hyp_tokens"] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                            "ref_tokens"] + [self.tokenizer_indices["</s>"]]
                    else:  # mask_dict["masking"] == "ref"
                        input_ids = mask_dict["ref_tokens"] + [self.tokenizer_indices["<sep>"]] + mask_dict[
                            "hyp_tokens"] + [self.tokenizer_indices["</s>"]]

                attention_mask = len(input_ids) * [1]

                input_ids = resize(input_ids, self.max_seq_length, self.tokenizer_indices["</s>"])
                attention_mask = resize(attention_mask, self.max_seq_length, 1)
                label_str = self.tokenizer.decode(mask_dict["label_tokens"], skip_special_tokens=True)

                input_batch["input_ids"].append(input_ids)
                input_batch["attention_mask"].append(attention_mask)
                labels.append(label_str)
                mask_tags.append(mask_dict["masking"])

        return input_batch, labels, mask_tags, word_numbers

    def predict(self, text_pairs: dict):
        # text_pairs should be a list of dict, each dict being of the form:
        #{"hypothesis": str,
        # "reference": str,
        # "hyp_lang": str,
        # "ref_lang": str}

        preds = []
        all_labels = []
        all_mask_tags = []
        all_word_numbers = [] #number each text pairs produced

        for i in range(0, len(text_pairs), self.model_batch_size):

            batch = text_pairs[i: i+self.model_batch_size]
            input_batch, labels, mask_tags, word_numbers = self.preprocess_batch(batch)
            all_word_numbers = all_word_numbers + word_numbers
            all_labels = all_labels + labels
            all_mask_tags = all_mask_tags + mask_tags

            with torch.no_grad():
                input_ids, attention_mask = torch.tensor(input_batch["input_ids"]),  torch.tensor(input_batch["attention_mask"])
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

                # generate answers + compute answerability
                gen_text = self.tokenizer.batch_decode(
                    dict_generated_ids['sequences'],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                preds += gen_text

        assert len(preds) == len(all_labels)
        assert len(preds) == len(all_mask_tags)
        assert len(all_word_numbers) == len(text_pairs)
        assert sum(all_word_numbers) == len(preds)

        outputs = []
        for k in range(len(preds)):
            output = {
                "prediction": preds[k],
                "ground_truth": all_labels[k],
                "masking": all_mask_tags[k],
                "comparison_metrics": {}, #TODO: any perplexity-related metrics should be added here
                "POS_tag": "NA", #TODO
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
            categorized_outputs.append(outputs[l:r])

        return categorized_outputs

