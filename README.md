# Token QuestEval
Token QuestEval is a modification of QuestEval  (license and release details below.)
![GitHub](https://img.shields.io/github/license/ThomasScialom/QuestEval)
![release](https://img.shields.io/github/v/release/ThomasScialom/QuestEval)

Instead of using noun chunks from a text passage as ground-truth answers from which questions are generated, each token of the passage is used as a ground-truth answer, and the corresponding question is the original text passage with the token masked by a special token <mask>. 

## Overview 
This repo contains the codes to perform the following tasks:
1. Create dataset to train and evaluate the QA model.
2. Train the QA model (finetune a T5 model) with the training dataset from step 1.
3. Evaluate summaries using the trained QA model (Token QuestEval)
4. Compute the correlation scores of summary evaluation.

## Installation
Create a virtual environment and download the required packages, which are listed in `requirements.txt`. From the command line, do the following:
```
python3 -m venv token_questeval
source token_questeval/bin/activate
pip install -r requirements.txt
```

# `token_questeval.py` Pipeline
##  Overview
**Note: Presented below is an overview of the steps taken to compute a score. Details about each function mentioned will follow in subsequent sections.**

Below is an example of instantiating `Token_QuestEval` and using it on two pairs of texts. 
    
```
from questeval.token_questeval import Token_QuestEval
questeval = Token_QuestEval()

source_1 = "It is a cat. It jumps away."
prediction_1 = "To catch the bird, the cat leaped."

source_2 = "The bird flies, landing on the top of the oak tree."
prediction_2 = "The bird escaped to the top of the tree."

score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2]
)

print(score)

```
    
When we instantiate the class using `questeval = Token_QuestEval()`, models are being loaded using `_load_all_models` and `get_model`. Focus on [line 123](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L123) to make sure that you're loading a T5 model that is appropriately trained. If memory is an issue, feel free to delete other lines, or to make sure that `get_model` is only called on models that you need to use.


When `corpus_questeval` is called, the input is divided in batches and passed into a method called `_batch_questeval`, which is on [line 67](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L67). It is the method that outlines the steps taken to compute the scores for the input pair of texts.
1. `_texts2logs` is called to write information such as the input text itself, the created masked segments and the ground truth labels into log files. Log files are stored in the  `Token_QuestEval/questeval/logs` folder.
2. `_compute_question_answering` is called twice: one time to fill the masked segments created from the hypothesis with the source, and one time to do the inverse: fill the masked segments created from the source with the hypothesis.
3. `_compute_answer_similarity_scores` is called to compute the f1 score between the predicted text from the previous step and the ground truth label. This step is applied on all log files.
4. `_calculate_score_from_logs` is finally called to compute the Token_QuestEval score for the input text using the log files. 

##  Loading Log Files
The main method  of this step is `_texts2logs` at [line 187](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L187). It calls several methods as detailed below:
###### a) `_load_logs`
1. Hash the text and uses it as the filename of the log file that corresponds to the text. For example, we would hash *It is a cat. It jumps away.* and use the hash value as filename. See [line 187](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L187)
2. If the hash value has never been seen before (If we don't have a log file in our `logs` folder corresponding to the text), create the log at [line 223](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L223)
See that the log is a dictionary with the following keys and values:
    -  **type**: the type of text. If it's a hypothesis, it would store the string **hyp**. If it's a source, it would store the string **src**. With our example of *"It is a cat. It jumps away."*, it would be **src**.
    -  **text**: the string of the text itself: *"It is a cat. It jumps away."*
    -  **self**: it's an empty dictionary that will later store the masked segments and the ground truth labels that are generated from the text (the **text** just above)
    -  **asked**: it's an empty dictionary that will later store the masked segments and the ground truth labels that are generated from the text (the **text** just above)

###### b) `_get_question_answers`
1. For every log files, it retrieves the text by taking `log['text']`. 
2. For each text, `_get_qas` is called to generate masked segements and ground truth labels. 
a) **Tokenization method**:  The tokenization method used here is English spaCy, but other methods such as split by whitespace, would work as well. Change the tokenization method in the code itself at [line 253](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L253)
b)  **Sliding window size**: For the sake of illustration, the sliding window size has been set to 4, so for each masked token, we take 4 tokens to its left (if they exist) and 4 tokens to its right (if they exist). See the graphics below for illustration. The default value is 24 and can be changed in the initiation phase of the model: `questeval = Token_QuestEval(sliding_window_size = 4)`
<img src="https://github.com/YuLuLiu/Token_QuestEval/blob/main/README_images/masked_segment_creation.PNG" width="600">

At this step, the `log['self']` should contain the masked segments and the ground-truth labels like so:
  
```
"self": {
    "TOKEN": {
      "QG_hash=ThomasNLG/t5-qg_squad1-en": {
        "questions": [
          " <mask> is a cat .",
          "It <mask> a cat . It",
          "It is <mask> cat . It jumps",
          "It is a <mask> . It jumps away",
          "It is a cat <mask> It jumps away .",
          "is a cat . <mask> jumps away .",
          "a cat . It <mask> away .",
          "cat . It jumps <mask> .",
          ". It jumps away <mask> "
        ]
      },
      "answers": [
        {
          "text": "It",
          "pos_tag": "PRON"
        },
        {
          "text": "is",
          "pos_tag": "AUX"
        },
        {
          "text": "a",
          "pos_tag": "DET"
        },
        {
          "text": "cat",
          "pos_tag": "NOUN"
        },
        {
          "text": ".",
          "pos_tag": "PUNCT"
        },
        {
          "text": "It",
          "pos_tag": "PRON"
        },
        {
          "text": "jumps",
          "pos_tag": "VERB"
        },
        {
          "text": "away",
          "pos_tag": "ADV"
        },
        {
          "text": ".",
          "pos_tag": "PUNCT"
        }
      ]
    }
  }
```
##  Fill Mask
the main method of this step is `_compute_question_answering` at [line 335 ](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L335). It uses one text to fill the masked segments generated from the other text. To be more specific, it takes in as input two logs in the following manners: `_compute_question_answering(logs_1: Dict, logs_2: Dict, type_logs_1: str, type_logs_2: str) `
1. Take the masked segments with their corresponding ground truth labels from `log_2['self']` at [line 355](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L355).
2. Initialize `log_1['asked']` as an dictionary with masked segments as keys at [line 362](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L362). The value of each key is a dictionary, storing information about the masked segment (its key).
3. call `_predict_answers` and store the outputs of the method into the dictionaries from step 2. The keys and values are described below:
    -  **answer**: the predicted text of the fill-mask model
    -  **answerability**: deplecated feature, used to measure how possible it is for the model to perform the QA/fill-mask task
    -  **bartscore**: equivalent to perplexity, explained in more details in the section on `predict_answers`.
    -  **ground_truth**: a dictionary storing the ground truth label as a string, but also other information such as its POS tags. In later steps, it will also store the f1 score comparing the ground truth label with the predicted text from **answer** above.

At this step, a key-value pair in the dictionary that is `log_1['asked']` should look like the following. Note that **QA_hash** stores the name of the fill-mask model we've used.
```
" <mask> catch the bird ,": {
      "QA_hash=yliu337/sliding_window_token_both_ctx": {
        "answer": "You",
        "answerability": 1.0,
        "bartscore": 0.1121981477690791,
        "ground_truth": {
          "To": {
            "pos_tag": "PART",
            "f1": 0
          }
        }
      }
    }
```

### More details on `_predict_answers`
This methods takes the masked segments and the ground-truth labels from `log_2['self']`, and the text from `log_1['text']` (that we consider as context), and format them to feed into the T5 fill mask model. This step is at [line 328](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L328) where the masked segment is concatenated with the context. For example, using the above masked segment *"<mask> catch the bird"* and *"It is a cat. It jumps away."* as context, we have: 
```
formated input = "<mask> catch the bird <sep> It is a cat. It jumps away."
corresponding label = "To"
```
Note that beyond passing a list of formated inputs and a list of corresponding labels, we also pass in an attribute of the class called `self.bartscore` at [line 331](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L331). If we did `questeval = Token_QuestEval(bartscore = False)`, the method will not compute bartscore and set the value to zero. 

T5 Model `predict` method is illustrated below and can be found at [line 77 of utils.py](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/utils.py#L77)
<img src="https://github.com/YuLuLiu/Token_QuestEval/blob/main/README_images/T5_model.PNG" width="600">

## Computing F1 score for each masked segment
The method `_compute_answer_similarity_scores` can be found at [line 508](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L508) and simply compares the predicted text with the ground-truth label by computing F1 score.

## Computing Token_QuestEval metric score for each pair of source & hypothesis
The main method of this step is `_calculate_score_from_logs` at [line 397 ](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L397). It does the following:
1. Check attribute `self.doc_types`. If the attribute only contains **mask_src**, then we only consider masked segments created by masking parts of the source text. Same logic for **mask_hyp**. If both are there (they are by default), we consider all masked segments. We can define this attribute at class instantiation phase: `questeval = Token_QuestEval(doc_types = ('mask_src',))`
2. Check `self.list_scores`. If the attribute only contains **f1**, then we only consider F1 score when we are computing the final metric score. Same logic for **bartscore**. If both are there (they are by default), we consider both. can define this attribute at class instantiation phase: `questeval = Token_QuestEval(list_scores = ('f1',))`
3. Calling `_base_score` on [line 464](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L464). This step depends heavily on step 1 and 2 above. Consider the case where `self.list_scores = ('f1',)` and `self.doc_types = ('mask_src',)`. 
    - We go into the log where the text is the hypothesis, named `hyp_log`. `hyp_log['asked']` contains masked segments generated from masking parts of the source.
    - For each masked segment, take the F1 score. This is done by `_get_scores` on [line 419](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L419)
    - The final metric score for this pair of source & hypothesis is an average of the F1 scores at [line 504](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L504)
If `self.doc_types = ('mask_src', 'mask_hyp')`, a final average is taken at [line 415](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L415) between what we get at step 3 for **mask_src** and for **mask_hyp**.

    
# MT evaluation pipeline
    
## Download and prepare data
(same json format as above)
    
Download data: TODO
    
Prepare examples: TODO
    
## Fine-tune T5 model
    
TODO
    
## Predict scores on WMT metrics data
    
    
TODO
