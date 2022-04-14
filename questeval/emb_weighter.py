from transformers import T5EncoderModel
import torch
import json
import logging
import os
import random
from questeval.weighter_utils import *

t5_encoder = T5EncoderModel.from_pretrained("t5-base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "cuda":
    t5_encoder.cuda()

# get embedding using T5Encoder model
def get_encoding(hyp_tokens, ref_tokens):
    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)
    merged = hyp_tokens + [32101] + ref_tokens + [1]

    with torch.no_grad():
      model_outputs = t5_encoder(input_ids=torch.tensor([merged]).to(device))
      hidden_states = model_outputs.last_hidden_state
      #delete eos token
      hidden_states = hidden_states[0][:-1, :]

      #split to delete the separator token
      hyp_K = hidden_states[:hyp_len, :]
      ref_K = hidden_states[-ref_len:, :]

      embeddings = torch.cat((hyp_K, ref_K), dim=0)

    return embeddings

def preprocess(log, score_type, mode):
    with torch.no_grad():
        processed_log = process_log(log, score_type, mode)

        #flatten the list of lists of tokens we got from processing
        hyp_tokens = [item for sublist in processed_log["hyp_word_list"] for item in sublist]
        ref_tokens = [item for sublist in processed_log["ref_word_list"]  for item in sublist]

        #get T5 embedding, this is part of the features of a training example
        embeddings = get_encoding(hyp_tokens, ref_tokens)

        if mode == "train":
            #gold = the value we compare & compute loss against
            gold = torch.tensor(processed_log["gold_score"], dtype=torch.float32).to(device)

            #also part of the features of a training example
            hyp_scores = torch.tensor(processed_log["hyp_scores"], dtype=torch.float32).to(device)
            ref_scores = torch.tensor(processed_log["ref_scores"], dtype=torch.float32).to(device)

    # concatenate hyp and ref representation
    if mode == "train":
        return {
            "embeddings": embeddings,
            "hyp_word_list": processed_log["hyp_word_list"],
            "ref_word_list": processed_log["ref_word_list"],
            "hyp_scores": hyp_scores,
            "ref_scores": ref_scores,
            "gold": gold
        }
    else:
        return {
            "embeddings": embeddings,
            "hyp_word_list": processed_log["hyp_word_list"],
            "ref_word_list": processed_log["ref_word_list"],
        }

class Emb_Weighter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 1, bias = False)
        self.softmax_fct = torch.nn.Softmax(dim=0)
        self.loss_fct = torch.nn.MSELoss()

    def forward(self, example):
        #apply linear layer to get weights: each token has its weight
        z =  self.linear(example["embeddings"])

        #reshape score z
        hyp_len = sum([len(word) for word in example["hyp_word_list"]])
        hyp_z = z[:hyp_len]
        ref_z = z[-(z.shape[0] - hyp_len):]

        #softmax so weights sum up to 1
        hyp_w = self.softmax_fct(hyp_z)
        ref_w = self.softmax_fct(ref_z)

        #get the predicted score by computed weighted sum of the prediction eval scores.
        mask_hyp_score = torch.matmul(example["hyp_scores"], hyp_w)
        mask_ref_score = torch.matmul(example["ref_scores"], ref_w)
        cat = torch.cat((mask_hyp_score, mask_ref_score))
        predicted_total_score = torch.mean(cat)
        return predicted_total_score

    def compute_weights(self, log):
        #get weights
        features = preprocess(log, "exact_match", "test")
        z =  self.linear(features["embeddings"])
        scores = z.reshape(z.shape[0]).tolist()

        #reshape score z
        hyp_len = sum([len(word) for word in features["hyp_word_list"]])
        hyp_z = z[:hyp_len]
        ref_z = z[-(z.shape[0] - hyp_len):]

        #softmax so weights sum up to 1
        hyp_w = self.softmax_fct(hyp_z)
        ref_w = self.softmax_fct(ref_z)
        weights = hyp_w.reshape(hyp_w.shape[0]).tolist() + ref_w.reshape(ref_w.shape[0]).tolist()

        #reshape into lists
        outputs = []
        for word in (features["hyp_word_list"] + features["ref_word_list"]):
            score_by_word = []
            weight_by_word = []
            for i in range(len(word)):
                score_by_word.append(scores.pop(0))
                weight_by_word.append(weights.pop(0))
            outputs.append({
                "weight": weight_by_word,
                "score": score_by_word}
            )
        return outputs

def main():
    logging.basicConfig(level=logging.DEBUG)

    # Get the training set
    logs = []
    # open the folder with all the training logs.
    # each log = a pair of source document - hypothesis
    for filename in os.listdir("questeval/EN_train_logs"):
        with open(os.path.join("questeval/EN_train_logs", filename), 'r') as f:
            log = json.load(f)
            logs.append(log)

    weighter = Emb_Weighter()
    if device == "cuda":
        weighter.cuda()

    dataset = [preprocess(log, "exact_match", "train") for log in logs]
    random.shuffle(dataset)
    train_dataset = dataset[:600]
    eval_dataset = dataset[-(len(dataset) - 600):]
    logging.debug('Training Set Processed')

    #initialize training
    optimizer = torch.optim.Adam(weighter.parameters(),
                                 lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    n_epochs = 101

    #start training
    print("Start training...\n")
    for epoch_i in range(n_epochs):
        total_loss = 0
        weighter.train()
        random.shuffle(train_dataset)
        for e in train_dataset:
            optimizer.zero_grad()

            # forward pass
            score = weighter.forward(e)

            # Compute loss and accumulate the loss values
            loss = loss_fn(score, e["gold"])
            total_loss += loss.item()

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weighter.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()


        eval_loss = 0
        for e in eval_dataset:
            with torch.no_grad():
                score = weighter.forward(e)
                loss = loss_fn(score, e["gold"])
                eval_loss += loss.item()

        #checkpoint stored and evaluated against eval set every epoch, change if wanted
        torch.save(weighter.state_dict(), ('questeval/emb_base_weighter.pt-' + str(epoch_i)))
        print('epoch {}, evaluation loss {}'.format(epoch_i, eval_loss))
        print('epoch {}, training loss {}'.format(epoch_i, total_loss))
    logging.debug("Training complete.")

if __name__ == "__main__":
    main()


