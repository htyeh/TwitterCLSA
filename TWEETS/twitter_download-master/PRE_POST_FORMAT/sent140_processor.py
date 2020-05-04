with open('sent140_train_raw.tsv') as test_input, open('sent140_train.tsv', 'w') as test_output:
    for line in test_input:
        split_line = line.split('\t')
        if split_line[0] == '0':
            polarity = 'Negative'
        elif split_line[0] == '2':
            polarity = 'Neutral'
        elif split_line[0] == '4':
            polarity = 'Positive'
        res_line = split_line[1] + '\t' + polarity + '\t' + split_line[-1]
        test_output.write(res_line)
