#!/usr/bin/python
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json_file')
args = parser.parse_args()

ref_len_total = 0
hyp_len_total = 0
num_total = 0

with open(args.json_file) as fp:
    for line in fp:
        jsonline = json.loads(line)
        ref_len_total += len(jsonline['reference'].split())
        hyp_len_total += len(jsonline['hypothesis'].split())
        num_total += 1

print('Ave ref len = ', ref_len_total/num_total)
print('Ave hyp len = ', hyp_len_total/num_total)
