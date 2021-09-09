import logging
import csv
import json
import spacy
from spacy.cli import download
import random
from datasets import load_dataset


def get_qas(list_sentences):
    SEP = ' '
    PADDING_SIZE = 24
    qas = []

    tokenized_sentences = [sentence.split() for sentence in list_sentences]

    for i in range(len(tokenized_sentences)):
        tokens = tokenized_sentences[i]
        index = random.randint(0, len(tokens) - 1)
        asw_token = tokens[index]
        mask_text = SEP.join(tokens[:index]) + ' <mask> ' + ' '.join(tokens[index + 1:])

        question = mask_text

        if PADDING_SIZE > 0:
            if i - 1 >= 0:
                left_neighbor = tokenized_sentences[i - 1]
                question = SEP.join(left_neighbor[-PADDING_SIZE:]) + SEP + question

            if i + 1 < len(tokenized_sentences):
                right_neighbor = tokenized_sentences[i + 1]
                question = question + SEP + SEP.join(right_neighbor[:PADDING_SIZE])

        QA_pair = {'answer': asw_token, 'question': question}
        qas.append(QA_pair)

    return qas

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


def sentencize(
    text: str, spacy_pipeline
):
    preprocessed_context = spacy_pipeline(text)
    return [sentence_tuple[0] for sentence_tuple in split_on_punct(preprocessed_context)]

def main():
    # get the sentence splitter
    download('en_core_web_sm')
    spacy_pipeline = spacy.load('en_core_web_sm')

    #get the dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = dataset['validation']

    MAX = 5000
    counter = 0
    
    #writing new dataset
    with open("cnn_token_ctx_article.jsonl", mode='w', encoding='utf-8') as outfile:
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
