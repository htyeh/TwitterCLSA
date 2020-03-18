#!/usr/local/bin/python3

neg_lines = []
neu_lines = []
pos_lines = []
with open('Mozetic_DE.tsv') as all_ids:
    for line in all_ids:
        res = [ele.strip() for ele in line.split()]
        formatted_line = res[0] + '\t' + res[1] + '\n'
        if res[1] == 'Negative':
            neg_lines.append(formatted_line)
        elif res[1] == 'Neutral':
            neu_lines.append(formatted_line)
        elif res[1] == 'Positive':
            pos_lines.append(formatted_line)

with open('mozetic_de_neg.tsv', 'w') as neg_ids:
    for line in neg_lines:
        neg_ids.write(line)
with open('mozetic_de_neu.tsv', 'w') as neu_ids:
    for line in neu_lines:
        neu_ids.write(line)
with open('mozetic_de_pos.tsv', 'w') as pos_ids:
    for line in pos_lines:
        pos_ids.write(line)

print(len(neg_lines))
print(len(neu_lines))
print(len(pos_lines))