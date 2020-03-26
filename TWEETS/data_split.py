#!/usr/local/bin/python3

# split full.tsv into train/dev/test (70/15/15)
# take last 30% for dev/test

import os

full_files = [file for file in os.listdir('.') if file.endswith('.tsv')]
print('splitting:', ', '.join(full_files))
for file in full_files:
    with open(file) as full_file:
        all_lines = full_file.readlines()
        dev_test_split = round(len(all_lines) * 0.15)
        train_split = len(all_lines) - 2 * dev_test_split
        train_lines = all_lines[:train_split]
        dev_lines = all_lines[train_split:train_split + dev_test_split]
        test_lines = all_lines[train_split + dev_test_split:]
    with open('train_' + file, 'w') as train_file:
        for line in train_lines:
            train_file.write(line)
    with open('dev_' + file, 'w') as dev_file:
        for line in dev_lines:
            dev_file.write(line)
    with open('test_' + file, 'w') as test_file:
        for line in test_lines:
            test_file.write(line)