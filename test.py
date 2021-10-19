from questeval.token_questeval import Token_QuestEval

def main():
    questeval = Token_QuestEval()
    prediction_1 = "Jean prefers apples over pears. Apples are sweeter."
    source_1 = "Jean likes apples more than pears. Pears are not that sweet."

    prediction_2 = "I want to take a nap at home. Let's go home to nap."
    source_2 = "I want to go home so I can take a nap. Let's go now."

    score = questeval.corpus_questeval(hypothesis=[prediction_1, prediction_2], sources=[source_1, source_2])
    print(score)

if __name__ == "__main__":
    main()
