import logging
import os
import pickle
from multiprocessing import Pool
from os import truncate
from typing import Tuple

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from datasets import Dataset as HFDataset


logger = logging.getLogger(__name__)


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

def preprocess_for_pred(batch, tokenizer, args):
    #batch a list of dicts, each dict has 'context' and 'qas'

    output = {"input_ids": [], "attention_mask": []}

    for example in batch:
        context = example['context']
        qas = example['qas']
        for qa in qas:
            formated_input = qa['question'] + ' <sep> ' + context
            tokenized_res = tokenizer(formated_input, max_length=args.max_seq_length, truncation=True)

            input_ids = resize(tokenized_res['input_ids'], args.max_seq_length)
            attention_mask = resize(tokenized_res['attention_mask'], args.max_seq_length)

            output["input_ids"].append(input_ids)
            output["attention_mask"].append(attention_mask)

    return output

def preprocess_data(data):
    example, tokenizer, args = data
    output = []

    context = example['context']
    tokenized_context = tokenizer(context, max_length=args.max_seq_length, truncation=True)['input_ids'][:-1]

    qas = example['qas']
    for qa in qas:

        formated_input = qa['question'] + ' <sep> ' + context
        tokenized_res = tokenizer(formated_input, max_length=args.max_seq_length, truncation=True)

        input_ids = resize(tokenized_res['input_ids'], args.max_seq_length)
        attention_mask = resize(tokenized_res['attention_mask'], args.max_seq_length)
        labels = tokenizer(qa['answer'], max_length=args.max_seq_length, truncation=True)['input_ids']
        labels = resize(labels, args.max_seq_length)

        output.append((torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)))

    return output

class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [(example, tokenizer, args) for example in data]

            list_examples = [preprocess_data(d) for d in tqdm(data)]
            self.examples = [item for sublist in list_examples for item in sublist]

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
