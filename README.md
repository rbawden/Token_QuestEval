# Quick Notes for Usage
Generic Usage:
```
from questeval.token_questeval import Token_QuestEval
questeval = Token_QuestEval(doc_types=X, list_scores=Y)
```
- X is a tuple of either ('mask_src',), ('mask_hyp',), or by default ('mask_hyp', 'mask_src'). This decides which sort of masked segments you want to consider in your score computation. 'mask_src' refers to masked segmented created from masking the source document, same logic for 'mask_hyp.'
- Y is a tuple of either ('f1',), ('bartscore',), or by default ('bartscore', 'f1'). This decides which metric you want to consider for every pair (masked segment, predicted fill). 
- IMPORTANT NOTE: in BEAMetrics, only doc_types is separated into three different metrics. So running the correlation score computation runs doc_type = ('mask_src',), ('mask_hyp',), and ('mask_hyp', 'mask_src'). This is currently impossible to do for list_scores, which requires you to set it up manually before running the correlation score computation in BEAMetrics.

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
## 1/ Creating QA Dataset: [create_QA_dataset.py](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py)
##### Original dataset: 
- The file creates a masked question answering dataset from CNN-Dailymail, if you wish to use another dataset, modify line  [line 66-68](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L66) to get the dataset.
- The file is tailored to the format of CNN-Dailymail, if you wish to use another dataset, modify  [line 75-78](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L75). For each example, there should be a string to be used as context, and a string to be used to generate masked questions. 

##### How examples are generated: 
- **Context Length**: When a string is intended to be used as context, the code currently takes its 5 first sentences. This number can be changed at [line 81](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L81).
- **Masked Question Creation**: When a string is intended to be used to generate masked QA pairs, it is split into tokens by whitespace (change splitting strategy if working with languages like Chinese). Random token is chosen to be be masked in the sentence to create the masked question. This is done in [line 15-23](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L15)
  
```
e.g.: 
  String : "I love eating apples. I hope I can eat at least one every day. They are very delicious."
  Working with sentence 2 : "I hope I can eat at least one every day."
  Selected token / Answer : "eat"
  Masked question : "I hope I can <mask> at least one every day."
```
- **Masked Question Padding**: The padding size is set to be 24, it means that we will take 24 tokens from the right and 24 tokens from the left of the masked token, from the original string. This is done in [line 25-32](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L25).

```
e.g.:
  Padding Size = 4 (for the sake of example)
  String : "I love eating apples. I hope I can eat at least one every day. They are very delicious."
  Padded masked question: "I hope I can <mask> at least one every""
```
##### Quick changes to be made:
- Output dataset size: [line 70](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L70)
- Output dataset path: [line 74](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L74), the `QA_data` is an empty folder that can be used to store it.

## 2/ Training QA Model: [train_t5.py](https://github.com/YuLuLiu/Token_QuestEval/blob/main/train_t5.py)
The code uses a modified version of [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers). The modifications tailor the data processing step to the dataset we created above. There should only be small changes to make to the code:
- Change the path to the dataset, and the proportion of train-eval split on [lines 21-26](https://github.com/YuLuLiu/Token_QuestEval/blob/main/train_t5.py#L21) 
- Change T5 Model training configurations as need on [lines 32-43](https://github.com/YuLuLiu/Token_QuestEval/blob/main/train_t5.py#L32), the possible options are presented in [simpletransformers documentation](https://simpletransformers.ai/docs/usage/)

## 3/ Token QuestEval: [/questeval/token_questeval.py](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py) 
The usage of Token QuestEval is identical to that of QuestEval, with the following points to consider:
-  `list_scores` can take a Tuple, combination of `f1`, `answerability` or `bartscore`, `bartscore` here measures the loss of the QA model given context, question and the ground-truth answer. Its implementation can be found [here](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/utils.py#L126) 
-  `self.filter_pos` if set to true, will only include QA pairs where the ground-truth answers are marked to be the wanted POS tag listed in  `self.wanted_pos`. 
-  `self.filter_answ` if set to true, will only include QA pairs where the ground-truth answers are NOT one of the stopwords listed in `self.stopwords`. There four attributes can be found on lines [lines 53-57](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L53) 

```
from questeval.token_questeval import Token_QuestEval
questeval = Token_QuestEval()

source_1 = "Since 2000, the recipient of the Kate Greenaway medal has also been presented with the Colin Mears award to the value of 35000."
prediction_1 = "Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Mears award of the Kate Greenaway medal."
references_1 = [
    "Since 2000, the recipient of the Kate Greenaway Medal will also receive the Colin Mears Awad which worth 5000 pounds",
    "Since 2000, the recipient of the Kate Greenaway Medal has also been given the Colin Mears Award."
]

source_2 = "He is also a member of another Jungiery boyband 183 Club."
prediction_2 = "He also has another Jungiery Boyband 183 club."
references_2 = [
    "He's also a member of another Jungiery boyband, 183 Club.", 
    "He belonged to the Jungiery boyband 183 Club."
]

score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2],
    list_references=[references_1, references_2]
)

print(score)
```
**Don't forget to change the spacy pipeline if you want to work in another language than English**
  
## 4/ Computing Correlation Scores [run_all.py](https://github.com/YuLuLiu/Token_QuestEval/blob/main/run_all.py) 
No change has been made to the computation of correlation scores. For example, for CNN-Dailymail, run: `python run_all.py --dataset SummarizationCNNDM`.
**Don't forget to unpack the data as detailed [run_all.py](https://github.com/ThomasScialom/BEAMetrics)**
  
  
# `token_questeval.py` Pipeline
##  Overview
**Note: Presented below is an overview of the steps taken to compute a score. Details about each function mentioned will follow in subsequent sections.**

Below is an example of instantiating `Token_QuestEval` and using it on two pairs of texts. 
```
from questeval.token_questeval import Token_QuestEval
questeval = Token_QuestEval()

source_1 = "The cat jumped over the fence to chase after the bird."
prediction_1 = "To catch the bird, the cat leaped over the fence."

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
1. Hash the text and uses it as the filename of the log file that corresponds to the text. For example, we would hash *"The cat jumped over the fence to chase after the bird."* and use the hash value as filename. See [line 187](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L187)
2. If the hash value has never been seen before (If we don't have a log file in our `logs` folder corresponding to the text), create the log at [line 223](https://github.com/YuLuLiu/Token_QuestEval/blob/main/questeval/token_questeval.py#L223)
See that the log is a dictionary with the following keys and values:
    -  **type**: the type of text. If it's a hypothesis, it would store the string **hyp**. If it's a source, it would store the string **src**. With our example of *"The cat jumped over the fence to chase after the bird."*, it would be **src**.
    -  **text**: the string of the text itself: *"The cat jumped over the fence to chase after the bird."*
    -  **self**: it's an empty dictionary that will later store the masked segments and the ground truth labels that are generated from the text (the **text** just above)
    -  **asked**: it's an empty dictionary that will later store the masked segments and the ground truth labels that are generated from the text (the **text** just above)

###### b) `_get_question_answers`
1. For every log files, it retrieves the text by taking `log['text']`. 
2. For each text, `_get_qas` is called to generate masked segements and ground truth labels. 
<img src="https://github.com/YuLuLiu/Token_QuestEval/blob/main/README_images/masked_segment_creation.PNG" width="200">
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
