# Token QuestEval
Token QuestEval is a modification of QuestEval  (license and release details below.)
![GitHub](https://img.shields.io/github/license/ThomasScialom/QuestEval)
![release](https://img.shields.io/github/v/release/ThomasScialom/QuestEval)

Instead of using noun chunks from a text passage as ground-truth answers from which questions are generated, each token of the passage is used as a ground-truth answer, and the corresponding question is the original text passage with the token masked by a special token <mask>. 

## Overview 
This repo contains the codes to perform the following tasks:
1. Create training dataset to train the QA model.
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
- **Masked Question Creation**: When a string is intended to be used to generate masked QA pairs, it is first split into sentences and then split into tokens by whitespace (change splitting strategy if working with languages like Chinese). For each sentence, a random token is chosen to be the answer, and the token is masked in the sentence to create the masked question. This is done in [line 15-23](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L15)
  
*e.g.: 
  
String : "I love eating apples. I hope I can eat at least one every day. They are very delicious."
  
Working with sentence 2 : "I hope I can eat at least one every day."
  
Selected token / Answer : "eat"
  
Masked question : "I hope I can <mask> at least one every day."*

- **Masked Question Padding**: The padding size is set to be 24, it means that we will take 24 tokens from the right and 24 tokens from the left of the masked question, from the original string. This is done in [line 25-32](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L25).

*e.g.:
  
Padding Size = 2 (for the sake of example)
  
String : "I love eating apples. I hope I can eat at least one every day. They are very delicious."
  
Masked question : "I hope I can <mask> at least one every day."
  
Padded masked question: "eating apples. I hope I can <mask> at least one every day. They are"*

##### Quick changes to be made:
- Output dataset size: [line 70](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L70)
- Output dataset name: [line 74](https://github.com/YuLuLiu/Token_QuestEval/blob/main/create_QA_dataset.py#L74)
