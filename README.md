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
## 1/ Creating QA Training Dataset

(WIP)
