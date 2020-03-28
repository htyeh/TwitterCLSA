#!/usr/local/bin/python3
import random

# randomly oversample neg
sampled_neg_lines = []
with open('neg.tsv') as neg:
    neg_lines = neg.readlines()
random.shuffle(neg_lines)

for i in range(7864):
    sampled_neg_lines.append(neg_lines[i])

with open('sampled_neg.tsv', 'w') as oversampled:
    for line in sampled_neg_lines:
        oversampled.write(line)

# randomly oversample pos
sampled_pos_lines = []
with open('pos.tsv') as pos:
    pos_lines = pos.readlines()
random.shuffle(pos_lines)

for i in range(7175):
    sampled_pos_lines.append(pos_lines[i])

with open('sampled_pos.tsv', 'w') as oversampled:
    for line in sampled_pos_lines:
        oversampled.write(line)