import logging
import csv
import json
import spacy
from spacy.cli import download
import random
from datasets import load_dataset


def get_masked_sequences(article, highlights):
      examples = []
      article_tokens = article.split()[:250]
      highlights_tokens = highlights.split()
      tokens = article_tokens + ['<sep>'] + highlights_tokens

      for i in range(len(tokens)):
          if tokens[i] != '<sep>':
              label = tokens[i]
              masked = ' '.join(tokens[0:i]) + ' <mask> ' + ' '.join(tokens[i + 1:])
              pair = {'answer': label, 'question': masked}
              examples.append(pair)

      random_indices = random.sample(range(len(examples)), min(10, len(examples)))
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
    train_dataset = dataset['train']

    MAX = 5000
    counter = 0
    
    #writing new dataset
    with open("weight_cnn_token.jsonl", mode='w', encoding='utf-8') as outfile:
        for e in train_dataset:
            counter = counter + 1
            highlights = e['highlights']
            article = e['article']

            qas = get_masked_sequences(article, highlights)

            json_record = json.dumps({'id': counter, 'context': ' ', 'qas': qas}, ensure_ascii=False)
            outfile.write(json_record + '\n')

            if counter > MAX:
                break


if __name__ == "__main__":
    main()
