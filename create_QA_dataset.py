import logging
from simpletransformers.ner import NERModel, NERArgs
import csv
import json
import spacy
from spacy.cli import download
import random

def get_qas(list_sentences):
    SEP = ' '
    qas = []
    for text in list_sentences:
        tokens = text.split()
        index = random.randint(0, len(tokens) - 1)
        asw_token = tokens[index]
        mask_text = SEP.join(tokens[:index]) + ' <mask> ' + ' '.join(tokens[index + 1:])

        QA_pair = {'answer': asw_token, 'question': mask_text}
        qas.append(QA_pair)

    return qas

def split_on_punct(doc):
    """
    From one spacy doc to a List of (sentence_text, (start, end))
    """
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


def sentencize(
    text: str, spacy_pipeline
):
    preprocessed_context = spacy_pipeline(text)
    return [sentence_tuple[0] for sentence_tuple in split_on_punct(preprocessed_context)]

def main():
    print("get the sentence splitter")
    download('en_core_web_sm')
    spacy_pipeline = spacy.load('en_core_web_sm')

    print("get the CNN dataset")
    from datasets import load_dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = dataset['validation']

    MAX = 5000
    counter = 0
    
    print("writing new dataset")
    with open("EVAL_cnn_token_ctx_article.jsonl", mode='w', encoding='utf-8') as outfile:
        for e in train_dataset:
            counter = counter + 1
            highlights = e['highlights']
            article = e['article']

            highlights_sents = sentencize(highlights, spacy_pipeline)
            article_sents = sentencize(article, spacy_pipeline)[:5]

            context = ' '.join(article_sents)
            qas = get_qas(highlights_sents)

            json_record = json.dumps({'id': counter, 'context': context, 'qas': qas}, ensure_ascii=False)
            outfile.write(json_record + '\n')

            if counter > MAX:
                break


if __name__ == "__main__":
    main()
