#!/usr/local/bin/python3
import random

# randomly upsampling neg
# long_lines = []
# sampled_lines = []
# with open('neg.tsv') as neg:
#     for line in neg:
#         res = [ele.strip() for ele in line.split('\t')]
#         if len(res[2].split()) >= 10:
#             to_append = res[0] + '\t' + res[1] + '\t' + res[2] + '\n'
#             long_lines.append(to_append)

# random.shuffle(long_lines)
# for i in range(2311):
#     sampled_lines.append(long_lines[i])

# with open('rand_neg.tsv', 'w') as oversampled:
#     for line in sampled_lines:
#         oversampled.write(line)

# randomly downsampling neu/pos

# long_lines = []
# sampled_lines = []
# with open('neu.tsv') as neu:
#     for line in neu:
#         res = [ele.strip() for ele in line.split('\t')]
#         if len(res[2].split()) >= 10:
#             to_append = res[0] + '\t' + res[1] + '\t' + res[2] + '\n'
#             long_lines.append(to_append)

# random.shuffle(long_lines)
# for i in range(10000):
#     sampled_lines.append(long_lines[i])

# with open('rand_neu.tsv', 'w') as downsampled:
#     for line in sampled_lines:
#         downsampled.write(line)

# randomly downsampling neu/pos

long_lines = []
sampled_lines = []
with open('pos.tsv') as pos:
    for line in pos:
        res = [ele.strip() for ele in line.split('\t')]
        if len(res[2].split()) >= 10:
            to_append = res[0] + '\t' + res[1] + '\t' + res[2] + '\n'
            long_lines.append(to_append)

random.shuffle(long_lines)
for i in range(10000):
    sampled_lines.append(long_lines[i])

with open('rand_pos.tsv', 'w') as downsampled:
    for line in sampled_lines:
        downsampled.write(line)