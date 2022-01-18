import logging
import os
import pickle
from multiprocessing import Pool
from os import truncate
from typing import Tuple
import re
import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from datasets import Dataset as HFDataset
import random
import spacy
from spacy.lang.id import Indonesian
from spacy.lang.tr import Turkish

logger = logging.getLogger(__name__)

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
        logging.warning("Downloading spaCy language model: " + pipeline_name)
        from spacy.cli import download
        download(pipeline_name)
        pipeline = spacy.load(pipeline_name)
    SPACY_PIPELINES[lang] = pipeline

SPACY_PIPELINES["id"] = Indonesian()
SPACY_PIPELINES["tr"] = Turkish()

def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    if args.preprocess_inputs:
        return tokenizer.prepare_seq2seq_batch(
            src_texts=[
                prefix + ": " + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ],
            tgt_texts=dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
    else:
        return tokenizer.prepare_seq2seq_batch(
            src_texts=[
                prefix + input_text
                for prefix, input_text in zip(dataset["prefix"], dataset["input_text"])
            ],
            tgt_texts=dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
def load_hf_dataset(data, tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
        )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=True,
    )

    dataset.set_format(type="pt", columns=["input_ids", "attention_mask"])

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset

def resize(original_list, desired_len):
    if len(original_list) > desired_len:
        output = original_list[:desired_len]
    elif len(original_list) < desired_len:
        output = original_list[:desired_len] + [0]*(desired_len-len(original_list))
    else:
        output = original_list

    return output

def get_word_list(sentence, t5_tokenizer, spacy_tokenizer):
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

def get_all_masked_sequences(hypothesis, reference, hyp_lang, ref_lang, t5_tokenizer, args):
    #get the list of words (intersection of spaCy and T5 tokenization)
    hyp_word_list = get_word_list(hypothesis, t5_tokenizer, SPACY_PIPELINES[hyp_lang])
    ref_word_list = get_word_list(reference, t5_tokenizer, SPACY_PIPELINES[ref_lang])

    ref_tokens = [item for sublist in ref_word_list for item in sublist]
    hyp_tokens = [item for sublist in hyp_word_list for item in sublist]

    all_masked_seqs = []

    #loop over all words in hypothesis
    for index in range(len(hyp_word_list)):
        label = [args.tokenizer_indices["<extra_id_0>"]] + hyp_word_list[index] \
                  + [args.tokenizer_indices["<extra_id_1>"], args.tokenizer_indices["</s>"]]

        copy_hyp_word_list = hyp_word_list.copy()
        copy_hyp_word_list[index] = args.tokenizer_indices["<extra_id_0>"]

        #get list of tokens before and after the masked word
        word_list_before = copy_hyp_word_list[0:index]
        token_list_before = [item for sublist in word_list_before for item in sublist]
        word_list_after = copy_hyp_word_list[index+1:]
        token_list_after = [item for sublist in word_list_after for item in sublist]

        #apply sliding window
        token_list_before = token_list_before[-args.sliding_window_size:]
        token_list_after = token_list_after[:args.sliding_window_size]
        masked_seq = token_list_before + [args.tokenizer_indices["<extra_id_0>"]] + token_list_after

        #compute nb of tokens left for context
        ref_length = args.max_seq_length - len(masked_seq) - 2
        ref_tokens = ref_tokens[:ref_length]

        mask_dict = {"masking": "hyp",
                     "label_tokens": label,
                     "hyp_tokens": masked_seq,
                     "ref_tokens": ref_tokens}

        all_masked_seqs.append(mask_dict)

    # loop over all words in reference
    for index in range(len(ref_word_list)):
        label = [args.tokenizer_indices["<extra_id_0>"]] + ref_word_list[index] \
                + [args.tokenizer_indices["<extra_id_1>"], args.tokenizer_indices["</s>"]]

        copy_ref_word_list = ref_word_list.copy()
        copy_ref_word_list[index] = args.tokenizer_indices["<extra_id_0>"]

        # get list of tokens before and after the masked word
        word_list_before = copy_ref_word_list[0:index]
        token_list_before = [item for sublist in word_list_before for item in sublist]
        word_list_after = copy_ref_word_list[index + 1:]
        token_list_after = [item for sublist in word_list_after for item in sublist]

        # apply sliding window
        token_list_before = token_list_before[-args.sliding_window_size:]
        token_list_after = token_list_after[:args.sliding_window_size]
        masked_seq = token_list_before + [args.tokenizer_indices["<extra_id_0>"]] + token_list_after

        # compute nb of tokens left for context
        hyp_length = args.max_seq_length - len(masked_seq) - 2
        hyp_tokens = hyp_tokens[:hyp_length]

        mask_dict = {"masking": "ref",
                     "label_tokens": label,
                     "hyp_tokens": hyp_tokens,
                     "ref_tokens": masked_seq}

        all_masked_seqs.append(mask_dict)

    return all_masked_seqs

def get_masked_sequence(focus, context, focus_lang, context_lang, t5_tokenizer, args):
  #get the list of words (intersection of spaCy and T5 tokenization)
  focus_word_list = get_word_list(focus, t5_tokenizer, SPACY_PIPELINES[focus_lang])
  context_word_list = get_word_list(context, t5_tokenizer, SPACY_PIPELINES[context_lang])

  #select (randomly) the word to mask
  index = random.randint(0, len(focus_word_list)-1)
  label = [args.tokenizer_indices["<extra_id_0>"]] + focus_word_list[index] \
          + [args.tokenizer_indices["<extra_id_1>"], args.tokenizer_indices["</s>"]]
  focus_word_list[index] = args.tokenizer_indices["<extra_id_0>"]

  #get list of tokens before and after the masked word
  word_list_before = focus_word_list[0:index]
  token_list_before = [item for sublist in word_list_before for item in sublist]
  word_list_after = focus_word_list[index+1:]
  token_list_after = [item for sublist in word_list_after for item in sublist]

  #apply sliding window
  token_list_before = token_list_before[-args.sliding_window_size:]
  token_list_after = token_list_after[:args.sliding_window_size]
  focus_tokens = token_list_before + [args.tokenizer_indices["<extra_id_0>"]] + token_list_after

  #compute nb of tokens left for context
  focus_length = len(focus_tokens)
  context_length = args.max_seq_length - focus_length - 2
  context_tokens = [item for sublist in context_word_list for item in sublist]
  context_tokens = context_tokens[:context_length]

  return focus_tokens, context_tokens, label

def preprocess_for_pred(batch, tokenizer, args, order="hyp_first"):
    # set order = "focus-first" to do: focus <sep> context
    # set order = "hyp_first" to do: hypothesis <sep> reference/source
    input_batch = {
        "input_ids": [],
        "attention_mask": []
    }
    labels = []
    mask_tags = []

    for example in batch:
        all_masked_seqs = get_all_masked_sequences(hypothesis=example['hypothesis'],
                                                   reference=example['reference'],
                                                   hyp_lang=example["hyp_lang"],
                                                   ref_lang=example["ref_lang"],
                                                   t5_tokenizer = tokenizer,
                                                   args = args)
        for mask_dict in all_masked_seqs:
            if order == "hyp_first":
                input_ids = mask_dict["hyp_tokens"] + [args.tokenizer_indices["<sep>"]] + mask_dict["ref_tokens"] + [args.tokenizer_indices["</s>"]]
            else:
                if mask_dict["masking"] == "hyp":
                    input_ids = mask_dict["hyp_tokens"] + [args.tokenizer_indices["<sep>"]] + mask_dict["ref_tokens"] + [args.tokenizer_indices["</s>"]]
                else: #mask_dict["masking"] == "ref"
                    input_ids = mask_dict["ref_tokens"] + [args.tokenizer_indices["<sep>"]] + mask_dict["hyp_tokens"] + [args.tokenizer_indices["</s>"]]

            attention_mask = len(input_ids) * [1]

            input_ids = resize(input_ids, args.max_seq_length)
            attention_mask = resize(attention_mask, args.max_seq_length)
            label_str = tokenizer.decode(mask_dict["label_tokens"], skip_special_tokens=True)

            input_batch["input_ids"].append(input_ids)
            input_batch["attention_mask"].append(attention_mask)
            labels.append(label_str)
            mask_tags.append(mask_dict["masking"])

    return input_batch, labels, mask_tags

def preprocess_data(data, order="hyp_first"):
    # set order = "focus-first" to do: focus <sep> context
    # set order = "hyp_first" to do: hypothesis <sep> reference/source
    example, tokenizer, args = data

    if order == "hyp_first":
        # randomly decide whether hyp or ref is the focus/going to be masked
        if random.randint(0, 1) == 0:
            hyp_tokens, ref_tokens, label = get_masked_sequence(focus=example['hypothesis'],
                                                                context=example['reference'],
                                                                focus_lang=example['hyp_lang'],
                                                                context_lang=example['ref_lang'],
                                                                t5_tokenizer = tokenizer,
                                                                args = args)
        else:
            ref_tokens, hyp_tokens, label = get_masked_sequence(focus=example['reference'],
                                                                context=example['hypothesis'],
                                                                focus_lang=example['ref_lang'],
                                                                context_lang=example['hyp_lang'],
                                                                t5_tokenizer = tokenizer,
                                                                args = args)

        input_ids = hyp_tokens + [args.tokenizer_indices["<sep>"]] + ref_tokens + [args.tokenizer_indices["</s>"]]

    else:  # order = focus hyp_first
        if random.randint(0, 1) == 0:
            focus_tokens, context_tokens, label = get_masked_sequence(focus=example['hypothesis'],
                                                                      context=example['reference'],
                                                                      focus_lang=example['hyp_lang'],
                                                                      context_lang=example['ref_lang'],
                                                                      t5_tokenizer = tokenizer,
                                                                      args = args)
        else:
            focus_tokens, context_tokens, label = get_masked_sequence(focus=example['reference'],
                                                                      context=example['hypothesis'],
                                                                      focus_lang=example['ref_lang'],
                                                                      context_lang=example['hyp_lang'],
                                                                      t5_tokenizer = tokenizer,
                                                                      args = args)
        input_ids = focus_tokens + [args.tokenizer_indices["<sep>"]] + context_tokens + [args.tokenizer_indices["</s>"]]

    attention_mask = len(input_ids) * [1]

    # add padding
    input_ids = resize(input_ids, args.max_seq_length)
    attention_mask = resize(attention_mask, args.max_seq_length)
    label = resize(label, args.max_seq_length)
    return (torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label))

class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode, filename=""):
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_"
            + str(args.max_seq_length)
            + str(len(data))
            + filename.split('/')[-1]
        ) #args.model_name.replace("/", "_")

        # get the indices for special tokens
        args.tokenizer_indices = {
            "<extra_id_0>": tokenizer.convert_tokens_to_ids("<extra_id_0>"),
            "<extra_id_1>": tokenizer.convert_tokens_to_ids("<extra_id_1>"),
            "<sep>": tokenizer.convert_tokens_to_ids("<sep>"),
            "</s>": tokenizer.convert_tokens_to_ids("</s>")
        }

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                #random.shuffle(self.examples) # RB addition
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)
            logger.info(" To be output to %s", cached_features_file)

            data = [(example, tokenizer, args) for example in data]

            self.examples = [preprocess_data(d) for d in tqdm(data)]

            if not args.no_cache:
                logger.info(
                    " Saving features into cached file %s", cached_features_file
                )
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]