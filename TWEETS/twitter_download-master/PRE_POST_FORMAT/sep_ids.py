#!/usr/local/bin/python3

# input: tsv with all labels
# output: 3 tsv according to label

import argparse
import os

parser = argparse.ArgumentParser(description="divide file according to sentiment")
parser.add_argument("input", help="specify input tsv file")
args = parser.parse_args()

neg_lines = []
neu_lines = []
pos_lines = []
with open(args.input) as all_ids:
    for line in all_ids:
        res = [ele.strip() for ele in line.split('\t')]
        # formatted_line = res[0] + '\t' + res[1] + '\t' + res[2] + '\n'
        if res[1] == 'Negative':
            neg_lines.append(line)
        elif res[1] == 'Neutral':
            neu_lines.append(line)
        elif res[1] == 'Positive':
            pos_lines.append(line)

new_dir_name = args.input.upper()[:-4]
os.mkdir(new_dir_name)

with open(os.path.join(new_dir_name, 'neg.tsv'), 'w') as neg_ids:
    for line in neg_lines:
        neg_ids.write(line)
with open(os.path.join(new_dir_name, 'neu.tsv'), 'w') as neu_ids:
    for line in neu_lines:
        neu_ids.write(line)
with open(os.path.join(new_dir_name, 'pos.tsv'), 'w') as pos_ids:
    for line in pos_lines:
        pos_ids.write(line)

print('neg lines:', len(neg_lines))
print('neu lines:', len(neu_lines))
print('pos lines:', len(pos_lines))