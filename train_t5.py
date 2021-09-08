import logging
import json
import random
from simpletransformers.t5 import T5Model, T5Args

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data

def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    pos_data = load_jsonl('/home/mila/y/yu-lu.liu/train_t5/data/token_data/cnn_token_ctx_article.jsonl')
    inv_pos_data = load_jsonl('/home/mila/y/yu-lu.liu/train_t5/data/token_data/cnn_token_ctx_highlight.jsonl')


    train_data = pos_data[:40000] + inv_pos_data[:40000]
    eval_data = pos_data[-10000:] + inv_pos_data[-10000:]

    random.shuffle(train_data)	
    random.shuffle(eval_data)	

    # Configure the model
    model_args = T5Args()
    model_args.num_train_epochs = 3
    model_args.fp16 = False
    model_args.max_seq_length = 512
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 8
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 5000
    model_args.save_steps = 5000
    model_args.silent = True
    model_args.output_dir = "train_t5QA_outputs/"
    model_args.special_tokens_list = ['<mask>', '<sep>', '<unanswerable>']

    #define the model
    model = T5Model("t5", "t5-base", args=model_args)

    # Train the model
    model.train_model(train_data, eval_data=eval_data)

if __name__ == "__main__":
    main()
