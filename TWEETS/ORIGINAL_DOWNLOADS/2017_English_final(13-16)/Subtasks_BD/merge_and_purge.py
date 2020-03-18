#!/usr/local/bin/python3
import os

needed_pos = 362
needed_neg = 12299
found_pos = 0
found_neg = 0
no_id = 0
# id<TAB>topic<TAB>label<TAB>text
existing_tweets = []
with open('../../../EN_13-16.tsv') as existing_file:
    for line in existing_file:
        existing_tweets.append(line.split('\t')[0])

uniq_ids = []
uniq_lines = []
en_files = [file for file in os.listdir('.') if file.startswith('twitter')]
print(en_files)
for en_file in en_files:
    file = open(en_file)
    for line in file:
        res = line.split('\t')
        if len(res) >= 3 and res[0] not in existing_tweets and res[2] != 'neutral':
            existing_tweets.append(res[0])
            uniq_ids.append(res[0])
            uniq_lines.append(res[0] + '\t' + res[2] + '\t' + res[3].strip() + '\n')
    file.close()

negs = []
for line in uniq_lines:
    if line.split('\t')[1] == 'negative':
        negs.append(line)

with open('uniq_neg_BD.tsv', 'w') as uniq_neg:
    for line in negs:
        uniq_neg.write(line)

# uniq_tweets = []
# old_train_tweets = []
# old_devtest_tweets = []
# new_tweets = []
# with open('../../TW_DATA_CLEAN/EN_train.tsv') as old_train:
#     for line in old_train:
#         old_train_tweets.append(line.split()[0])
#         uniq_tweets.append(line.split()[0])
#     print('old_train: ' + str(len(old_train_tweets)))
#     print(len(set(old_train_tweets)))
# with open('../../TW_DATA_CLEAN/EN_dev+test.tsv') as old_devtest:
#     for line in old_devtest:
#         old_devtest_tweets.append(line.split()[0])
#         uniq_tweets.append(line.split()[0])
#     print('old_devtest: ' + str(len(old_devtest_tweets)))
#     print(len(set(old_devtest_tweets)))

# print(len(uniq_tweets))
# with open('SemEval13train+dev.tsv') as new_file:
#     for line in new_file:
#         new_tweets.append(line.split()[0])
#         uniq_tweets.append(line.split()[0])
#     print(len(new_tweets))
#     print(len(uniq_tweets))
#     print(len(set(uniq_tweets)))



# with open('DE.tsv', 'w') as file:
#     for line in uniq_lines:
#         file.write(line)