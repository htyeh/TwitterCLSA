# SENTIPOLC tweets are labeled with subjectivity and polarities, where both or no polaritiy can coexist
# field [0] = tw id
# field [2] = pos (0/1)
# field [3] = neg (0/1)
# field [-1] = text
# if field [2] and field [3] both 0 -> label as neutral
# if field [2] and field [3] both 1 -> discard

import sys

with open('sentipolc_train_raw.tsv') as train_input, open('sentipolc_train.tsv', 'w') as train_output:
    for line in train_input:
        split_line = line.split('\t')
        if len(split_line) == 9:
            if split_line[2] == '1' and split_line[3] == '0':
                polarity = 'Positive'
            elif split_line[2] == '0' and split_line[3] == '1':
                polarity = 'Negative'
            elif split_line[2] == '0' and split_line[3] == '0':
                polarity = 'Neutral'
            else:
                continue
            res_line = split_line[0] + '\t' + polarity + '\t' + split_line[-1]
            train_output.write(res_line)

with open('sentipolc_test_raw.tsv') as test_input, open('sentipolc_test.tsv', 'w') as test_output:
    for line in test_input:
        split_line = line.split('\t')
        if len(split_line) == 9:
            if split_line[2] == '1' and split_line[3] == '0':
                polarity = 'Positive'
            elif split_line[2] == '0' and split_line[3] == '1':
                polarity = 'Negative'
            elif split_line[2] == '0' and split_line[3] == '0':
                polarity = 'Neutral'
            else:
                continue
            res_line = split_line[0] + '\t' + polarity + '\t' + split_line[-1]
            test_output.write(res_line)