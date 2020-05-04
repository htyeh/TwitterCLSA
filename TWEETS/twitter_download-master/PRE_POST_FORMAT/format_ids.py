#!/usr/local/bin/python3

# input: tsv with all labels
# output: 3 tsv according to label

neg_lines = []
neu_lines = []
pos_lines = []
with open('sent140_train.tsv') as all_ids:
    for line in all_ids:
        res = [ele.strip() for ele in line.split('\t')]
        formatted_line = res[0] + '\t' + res[1] + '\t' + res[2] + '\n'
        if res[1] == 'Negative':
            neg_lines.append(formatted_line)
        elif res[1] == 'Neutral':
            neu_lines.append(formatted_line)
        elif res[1] == 'Positive':
            pos_lines.append(formatted_line)

with open('neg.tsv', 'w') as neg_ids:
    for line in neg_lines:
        neg_ids.write(line)
with open('neu.tsv', 'w') as neu_ids:
    for line in neu_lines:
        neu_ids.write(line)
with open('pos.tsv', 'w') as pos_ids:
    for line in pos_lines:
        pos_ids.write(line)

print(len(neg_lines))
print(len(neu_lines))
print(len(pos_lines))