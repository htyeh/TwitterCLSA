#!/usr/local/bin/python3
# remove tweets already in the test directory from train data

uniq_ids = []
test_ids = []
uniq_lines = []
with open('../test/pos.tsv') as file:
    for line in file:
        if line.split()[0] not in test_ids:
            test_ids.append(line.split()[0])
print(len(test_ids))
with open('pos.tsv') as input:
    for line in input:
        if line.split()[0] not in test_ids:
            uniq_ids.append(line.split()[0])
            uniq_lines.append(line)
with open('train_pos.tsv', 'w') as output:
    for line in uniq_lines:
        output.write(line)
