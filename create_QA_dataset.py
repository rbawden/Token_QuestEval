import logging
import csv
import json
import spacy
from spacy.cli import download
import random
import math
from datasets import load_dataset

def main():
    # define language
    language = "en"

    #get the dataset
    #dataset = load_dataset('mlsum', language)
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    dataset = dataset["train"]

    #writing new dataset
    MAX_per_doc = 10000
    nb_docs = math.ceil(len(dataset)/MAX_per_doc)
    counter = 0

    for i in range(nb_docs):
        filename = "QA_data/summ_train_data.jsonl-{language}-{doc_nb}".format(language=language, doc_nb=i)
        with open(filename, mode='w', encoding='utf-8') as outfile:
            data_subset = dataset[(i * MAX_per_doc):((i + 1) * MAX_per_doc)]
            for k in range(MAX_per_doc):
                id = language + "_" + str(counter)
                hypothesis = data_subset['highlights'][k]
                reference = data_subset['article'][k]

                json_record = json.dumps({'id': id, 'hypothesis': hypothesis, 'reference': reference,
                                          'hyp_lang': language, 'ref_lang': language}, ensure_ascii=False)
                outfile.write(json_record + '\n')
                counter = counter + 1

if __name__ == "__main__":
    main()
