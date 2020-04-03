#!/usr/local/bin/python3
import random

# randomly downsample neg/neu/pos to a certain # of Tweets

sampled_neg_lines = []
with open('neg.tsv') as neg:
    neg_lines = neg.readlines()
random.shuffle(neg_lines)

for i in range(1939):
    sampled_neg_lines.append(neg_lines[i])

with open('sampled_neg.tsv', 'w') as downsampled:
    for line in sampled_neg_lines:
        downsampled.write(line)

sampled_neu_lines = []
with open('neu.tsv') as neu:
    neu_lines = neu.readlines()
random.shuffle(neu_lines)

for i in range(5904):
    sampled_neu_lines.append(neu_lines[i])

with open('sampled_neu.tsv', 'w') as downsampled:
    for line in sampled_neu_lines:
        downsampled.write(line)

sampled_pos_lines = []
with open('pos.tsv') as pos:
    pos_lines = pos.readlines()
random.shuffle(pos_lines)

for i in range(2729):
    sampled_pos_lines.append(pos_lines[i])

with open('sampled_pos.tsv', 'w') as downsampled:
    for line in sampled_pos_lines:
        downsampled.write(line)
