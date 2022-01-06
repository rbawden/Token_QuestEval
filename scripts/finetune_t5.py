import logging
import json
import random
from simpletransformers.t5 import T5Model, T5Args
import os, glob
import torch

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def print_memory():

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    logging.info('Memory: %s reserved, %s allocated, %s free', r, a, f)

    
def train(data_filename, output_dir, nepochs=1, bsz=8, evalsteps=5000, savesteps=5000):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


    # Configure the model
    model_args = T5Args()
    model_args.num_train_epochs = nepochs
    model_args.fp16 = False
    model_args.max_seq_length = 512
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = bsz
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = evalsteps
    model_args.save_steps = savesteps
    model_args.no_cache = False
    model_args.reprocess_input_data = False
    model_args.silent = False
    model_args.output_dir = output_dir #"train_t5MT_outputs/"
    model_args.special_tokens_list = ['<mask>', '<sep>', '<unanswerable>']
    model_args.overwrite_output_dir = True

    # amount of validation data
    valsize = 2000
    
    # define the model or get last checkpoint (if eval has been done)
    filename = 't5-base'
    training_progress_file = ''
    eval_results_file = ''
    prev_checkpoints = glob.glob(model_args.output_dir + '/checkpoint-*')
    if len(prev_checkpoints) > 0:
        num = 0
        with open(model_args.output_dir + '/training_progress_scores.csv') as fp:
            evals = [x.split(',')[0] for x in fp.readlines()]
        for checkpoint in prev_checkpoints:
            new_num = checkpoint.split('-')[-1]
            if new_num in evals:
                num = max(num, int(new_num))
        if num != 0:
            filename = model_args.output_dir + '/checkpoint-' + str(num)
            training_progress_file = model_args.output_dir + '/training_progress_scores.csv'
            eval_results_file = model_args.output_dir + '/eval_results.txt'
    model = T5Model("t5", filename, args=model_args)

    print_memory()
    
    # Train the model
    os.sys.stderr.write('>> Fine-tuning with ' + data_filename + '\n')
    all_data = load_jsonl(data_filename)
    random.shuffle(all_data)
    train_data = all_data[:-valsize]
    eval_data = all_data[-valsize:]
    
    #random.shuffle(train_data)
    #random.shuffle(eval_data)	
    
    model_args.filename = data_filename.split('/')[-1]
    model.train_model(train_data, eval_data=eval_data, filename=model_args.filename,
                      training_progress_file=training_progress_file, eval_results_file=eval_results_file,
                      continue_progress=False)
    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_prefix')
    parser.add_argument('--output_dir', default='train_t5MT_outputs/')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--bsz', default=8, type=int)
    parser.add_argument('--evalsteps', default=5000, type=int)
    parser.add_argument('--savesteps', default=5000, type=int)
    args = parser.parse_args()

    os.sys.stderr.write('>> About to train\n')
    train(args.data_prefix, args.output_dir, args.epochs, args.bsz, args.evalsteps, args.savesteps)
    
