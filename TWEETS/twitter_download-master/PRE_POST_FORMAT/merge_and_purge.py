#!/usr/local/bin/python3

uniq_ids = []
available_lines = []
with open('downloaded.tsv') as downloaded:
    for line in downloaded:
        res = [ele.strip() for ele in line.split('\t')]
        if res[0] not in uniq_ids and res[2] != 'Not Available':
            uniq_ids.append(res[0])
            available_lines.append(line)
print(len(available_lines))

with open('mozetic_de_pos.tsv', 'w') as output:
    for line in available_lines:
        output.write(line)