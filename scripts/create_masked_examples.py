import logging
import csv
import json
import spacy
#from spacy.cli import download
import random
import os
import sys
#from datasets import load_dataset


def get_qas(hyp, src, padding_size = 24):
    examples = []

    tokens = hyp.split() + ['<sep>'] + src.split()
    for i in range(len(tokens)):
        if tokens[i] != '<sep>':
            label = tokens[i]
            masked = ' '.join(tokens[0:i]) + ' <mask> ' + ' '.join(tokens[i + 1:])
            pair = {'answer': label, 'question': masked}
            examples.append(pair)
            
        random_indices = random.sample(range(len(examples)), 1) # take one random
        output_examples = []
        for i in random_indices:
            output_examples.append(examples[i])
            
    return output_examples


def split_on_punct(doc):
    start = 0
    seen_period = False
    start_idx = 0
    for i, token in enumerate(doc):
        if seen_period and not token.is_punct:
            yield doc[start: token.i].text, (start_idx, token.idx)
            start = token.i
            start_idx = token.idx
            seen_period = False
        elif token.text in [".", "!", "?"]:
            seen_period = True
    if start < len(doc):
        yield doc[start: len(doc)].text, (start_idx, len(doc.text))


def sentencize(text: str, spacy_pipeline):
    preprocessed_context = spacy_pipeline(text)
    return [sentence_tuple[0] for sentence_tuple in split_on_punct(preprocessed_context)]


if __name__ == "__main__":

    i = 0
    for line in sys.stdin:
        if len(line.strip().split('\t')) != 2:
            continue
        h, s = line.strip().split('\t')
        h, s = s.strip(), h.strip()
        json_record = json.dumps({'id': i, 'context': '', 'qas': get_qas(h, s, padding_size=0)}, ensure_ascii=False)
        print(json_record)

