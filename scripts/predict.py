from questeval.token_questeval import Token_QuestEval
import sys

def process_stdin():
    questeval = Token_QuestEval(list_scores=('f1', 'bartscore'), use_cache=False)
    hs, rs = [], []
    for line in sys.stdin:
        h, r = line.strip().split('\t')
        h, r = h.strip(), r.strip()
        hs.append(h)
        rs.append(r)

    score = questeval.corpus_questeval(hypothesis=hs, sources=rs)
    print(score)


def process_files():
    questeval = Token_QuestEval()
    with open(hyp_file) as hf, open(ref_file) as rf:
        hyps = [x.strip() for x in hf.readlines()]
        refs = [x.strip() for x in rf.readlines()]
        
        score = questeval.corpus_questeval(hypothesis=hyps, sources=refs)
        print(score)

if __name__ == '__main__':

    process_stdin()
