import json
import sys


def process_all(args):
    i = 0
    # read from std input (tab separated lines, with hypothesis first)
    for line in sys.stdin:
        # if there are not two columns, then ignore (should not happen often)
        if len(line.strip().split('\t')) != 2:
            continue
        # get hypothesis and source/ref sentences
        h, s = line.strip().split('\t')
        h, s = s.strip(), h.strip()

        # maximum number of tokens = 150
        if len(h.split()) > 150 or len(s.split()) > 150:
            continue
        # output json record for this sentence
        json_record = json.dumps({'id': i, 'hypothesis': h, 'reference': s,
                                  'hyp_lang': args.hyp_lang, 'ref_lang': args.ref_lang}, ensure_ascii=False)
        print(json_record)
        i += 1

                
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_lang', default='en', type=str, help='Language of hypothesis (1st col)')
    parser.add_argument('--ref_lang', default='en', type=str, help='Language of reference/source (2nd col)')
    args = parser.parse_args()
    
    process_all(args)
