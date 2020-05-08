#!/usr/local/bin/python3
# purges repeating and Not Available lines
# python3 merge_and_purge.py downloaded.tsv purged.tsv
import argparse
parser = argparse.ArgumentParser(description="specify file to clean")
parser.add_argument("input", help="specify input tsv file")
parser.add_argument("output", help="specify output tsv file")
args = parser.parse_args()

uniq_ids = []
available_lines = []
with open(args.input) as downloaded:
    for line in downloaded:
        res = [ele.strip() for ele in line.split('\t')]
        if res[0] not in uniq_ids and res[2] != 'Not Available':
            uniq_ids.append(res[0])
            available_lines.append(line)
print(len(available_lines))

with open(args.output, 'w') as output:
    for line in available_lines:
        output.write(line)