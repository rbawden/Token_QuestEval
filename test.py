from questeval.maskeval import MaskEval

def main():
    questeval = MaskEval(fill_mask_model_name="models/t5_cnn_model/checkpoint-85920")
    hypothesis = ["I have trouble coming up with characters' names",
                  "This is a fun-loving copper-coated bear!",
                  "Our trip to Timor-Leste didn't cost us more than 2,000$.",
                  "It was snowing heavily today."
                  ]
    reference = ["It's difficult for me to name fictional characters",
                 "I saw a fun_loving bear today, covered with copper!",
                 "Our trip to Timor-Leste isn't expensive.",
                 "We had a snowy day."]

    score = questeval.corpus_questeval(hypothesis=hypothesis, references=reference)
    print(score)

if __name__ == "__main__":
    main()
