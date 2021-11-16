#!/usr/bin/python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def f_score():
    return

def bart_score():
    return

def predict_t5(model, hyp, source, batch_size=2):

    tokenizer = T5Tokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)

    input_text = []
    ref_toks = []
    with open(hyp) as hf, open(source) as sf:
        # accumulate examples
        for sid, (h, s) in enumerate(zip(hf, sf)):    
            # RB: check tokenisation
            h_tok = h.strip().split()
            for tok_id, tok in enumerate(h_tok):
                input_text.append(' '.join(h_tok[:tok_id]) + ' <mask> ' + ' '.join(h_tok[tok_id + 1:]) + ' <sep> ' + s.strip())
                example = {'sid': sid, 'tokid': tok_id, 'type': 'hyp', 'masked': tok, 'hyp': None, 'fscore': None, 'bartscore': None}
                ref_toks.append(example)
            s_tok = s.strip().split()
            for tok_id, tok in enumerate(h_tok):
                input_text.append(h.strip() + ' <sep> ' + ' '.join(h_tok[:tok_id]) + ' <mask> ' + ' '.join(h_tok[tok_id + 1:]))
                example = {'sid': sid, 'tokid': tok_id, 'type': 'source', 'masked': tok, 'hyp': None, 'fscore': None, 'bartscore': None}
                ref_toks.append(example)
                
        # split into batches and predict
        inputs = tokenizer(input_text, return_tensors='pt', padding=True)
        num_examples_seen = 0
        finished = False
        while not finished:
            print(input_text[num_examples_seen:num_examples_seen + batch_size])
            outputs = model.generate(
                input_ids=inputs['input_ids'][num_examples_seen:num_examples_seen + batch_size],
                attention_mask=inputs['attention_mask'][num_examples_seen:num_examples_seen + batch_size],
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
            output_toks = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
            num_examples_seen += batch_size

            # for each example, get scores
            for exid, ex_scores in enumerate(outputs.scores):
                print(ex_scores)
                print(ex_scores.shape)
                print(outputs.sequences[exid])
                print(outputs.sequences[exid].shape)
                output_scores = ex_scores[exid].gather(0, outputs.sequences[exid])
                output_scores = output_scores[outputs.sequences[exid] > 1]

                # logits
                example[num_examples_seen + exid]['logits'] = sum(output_scores) / len(output_scores)
                # correct or not
                example[num_examples_seen + exid]['acc'] = example[num_examples_seen + exid]['masked']
                print(example)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('hyp')
    parser.add_argument('source')
    args = parser.parse_args()

    
    predict_t5(args.model, args.hyp, args.source)
