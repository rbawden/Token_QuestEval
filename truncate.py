import logging
from transformers import T5TokenizerFast
import json

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    t5_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    filename = "summ_train_data.jsonl-en-0"
    data = load_jsonl(("QA_data/" + filename))
    truncated_data = []

    filename = ("QA_data/TRUNCATED_" + filename)
    with open(filename, mode='w', encoding='utf-8') as outfile:
        for d in data:
            trucated_hyp_ids = t5_tokenizer(d['hypothesis'], max_length=120, truncation=True)['input_ids']
            trucated_hyp = t5_tokenizer.decode(trucated_hyp_ids)

            trucated_src_ids = t5_tokenizer(d['reference'], max_length=380, truncation=True)['input_ids']
            trucated_src = t5_tokenizer.decode(trucated_src_ids)

            json_record = json.dumps({'id': d["id"], 'hypothesis': trucated_hyp, 'reference': trucated_src,
                                          'hyp_lang': d['hyp_lang'], 'ref_lang':  d['ref_lang']}, ensure_ascii=False)
            outfile.write(json_record + '\n')


if __name__ == "__main__":
    main()


