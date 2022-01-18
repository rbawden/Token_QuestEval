# MaskEval
MaskEval  is a modification of QuestEval  (license and release details below.)
![GitHub](https://img.shields.io/github/license/ThomasScialom/QuestEval)
![release](https://img.shields.io/github/v/release/ThomasScialom/QuestEval)

## Overview 
This repo contains the codes to perform the following tasks:
1. Create dataset to train and evaluate the fill-mask model.
2. Train the fill-mask model (finetune a T5 model) with the training dataset from step 1.
3. Evaluate summaries using the trained model (MaskEval)
4. Compute the correlation scores of summary evaluation.

## Installation
Create a virtual environment and download the required packages, which are listed in `requirements.txt`. From the command line, do the following:
```
python3 -m venv token_questeval
source token_questeval/bin/activate
pip install -r requirements.txt
```
## Quick Usage
Below is an example usage where MaskEval is instantiated with a model (whose path is to be specified by user) and called to evaluate two pairs of texts. Note that the length of hypothesis and that of references must be equal, as a one-to-one relationship is assumed. Both also need to be a list of strings.
```
from questeval.maskeval import MaskEval

maskeval = MaskEval(fill_mask_model_name = <INSERT PATH>)

hypothesis = ["It was snowing heavily today.",
            "Our trip to Timor-Leste didn't cost us more than 2,000$." ]
references = ["We had a snowy day.",
             "Our trip to Timor-Leste isn't expensive."]
             
score = maskeval.corpus_questeval(hypothesis=hypothesis, 
                                    references=references)
print(score)
```
Each pair generate a log file. Each log file is of the following form:
```
{
  "id_text": "It was snowing heavily today. <sep> We had a snowy day.",
  "hyp_text": "It was snowing heavily today.",
  "ref_text": "We had a snowy day.",
  "prediction_done": true,
  "answ_sim_computed": true,
  "weights_computed": true, 
  "masked": [
    {
      "prediction": "It",
      "ground_truth": "It",
      "masking": "hyp",
      "comparison_metrics": {
        "exact_match": 1
      },
      "POS_tag": "PRON",
      "weight": 0.23
    },
   ...]
   }
```
!!! TODO: POS tagging and weighter have yet to implemented, so `POS_tag = NA`, `weight = 0.0`, and `weights_computed = false` for now. 

----------------------WORK IN PROGRESS----------------------

    
# MT evaluation pipeline
    
## Download and prepare data
(same json format as above)
    
Download data and prepare training examples:

```
bash scripts/download_data.sh # downloads paraphrase data and MT data
bash scripts/create_paraphrase_data.sh # create paraphrase examples
bash scripts/create_mt_data.sh # create MT examples
```
    
Training files are:

- `data/paraphrase/parabank{1,2}-parts/*` (in several parts because there is a lot more data)
- `data/metrics/wmt14-18-intoEnglish-all.hyp-ref.masked-examples.jsonl`
    
    
## Fine-tune T5 model
    
Fine-tune on parabank1 data:
```
for i in {0..171}; do
    python -u scripts/finetune_t5.py data/paraphrase/parabank1-parts/parabank1.threshold0.7.detok.masked-examples.jsonl.part-$i \
                    --output_dir models/train_t5_parabank1/ 2>> models/train_t5_parabank1/train.log
done
```
(or use arrays of jobs in slurm)
    
Fine-tune on parabank2 data:
```
for i in {0..77}; do
    python -u scripts/finetune_t5.py data/paraphrase/parabank2-parts/parabank2.masked-examples.jsonl.part-$i \
                    --output_dir models/train_t5_parabank2/ 2>> models/train_t5_parabank2/train.log
done
```
(or use arrays of jobs in slurm)

Fine-tune on metrics data:
```
python scripts/finetune_t5.py data/metrics/wmt14-18-intoEnglish-all.hyp-ref.masked-examples.jsonl \
                   --epochs 5 \
                   --output_dir models/train_t5_metrics/ 2>> models/train_t5_metrics/train.log
```
    
## Predict scores on WMT metrics data
    
    
TODO
