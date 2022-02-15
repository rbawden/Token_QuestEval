import logging
from simpletransformers.t5 import T5Model, T5Args
import json

def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model_args = T5Args()
    model_args.max_seq_length = 512
    model_args.special_tokens_list = ['<sep>']
    model = T5Model(model_type = "t5",
                    model_name ="t5-base",
                    args=model_args)

    # Make predictions with the model

    example1 = {"hypothesis": "The elderly woman suffered from diabetes and hypertension, ship's doctors say .Previously, 86 passengers had fallen ill",
                "reference": "This is a source document.",
                "hyp_lang": "en",
                "ref_lang": "en"}

    to_predict = [example1, example1, example1, example1, example1]

    dataset = model.load_and_cache_examples(to_predict)

    filename = "t5_dataset.jsonl"
    with open(filename, mode='w', encoding='utf-8') as outfile:
        for k in range(len(dataset.examples)):
            json_record = json.dumps({"input_ids" : dataset.examples[k][0].tolist(), "label": dataset.examples[k][2].tolist()}, ensure_ascii=False)
            outfile.write(json_record + '\n')
    #assert len(preds) == len(labels)

    #filename = "t5_test_results.jsonl"
    #with open(filename, mode='w', encoding='utf-8') as outfile:
        #for k in range(len(preds)):
            #json_record = json.dumps({"pred" : preds[k], "ground_truth": labels[k], "mask_tag": mask_tags[k]}, ensure_ascii=False)
            #outfile.write(json_record + '\n')

if __name__ == "__main__":
    main()


