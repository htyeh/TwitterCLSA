#!/usr/local/bin/python3
# remove tweets already in the test directory from train
# remove tweets already in the val directory from train

uniq_ids = []
dev_ids = []
uniq_lines = []
with open('dev/neg.tsv') as file:
    for line in file:
        if line.split()[0] not in dev_ids:
            dev_ids.append(line.split()[0])
print(len(dev_ids))
with open('train/neg.tsv') as input:
    for line in input:
        if line.split()[0] not in dev_ids:
            uniq_ids.append(line.split()[0])
            uniq_lines.append(line)

with open('train_neg.tsv', 'w') as output:
    for line in uniq_lines:
        output.write(line)
