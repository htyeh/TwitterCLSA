#!/usr/local/bin/python3

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys

def load_embs_2_dict(path, dim=100):
    print('loading ' + path)
    embeddings_index = {}
    with open(path) as f:
        for line in tqdm(f.readlines(), desc="Loading", unit='embedding'):
            values = line.split()
            if len(values[1:]) == dim:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    print('found %s word vectors.' % len(embeddings_index))
    return embeddings_index

parser = argparse.ArgumentParser(description="calculate cross-lingual cossim using a dictionary")
parser.add_argument("src_embs", help="specify source embeddings")
parser.add_argument("tgt_embs", help="specify target embeddings")
parser.add_argument("-d", "--dict", help="specify dictionary", required=True)
args = parser.parse_args()

src_embs_index = load_embs_2_dict(args.src_embs)    # word:nparray
tgt_embs_index = load_embs_2_dict(args.tgt_embs)
with open(args.dict) as dict_file:
    word_pairs = {line.split()[0]: line.split()[1] for line in dict_file}   # src_word:tgt_word

src_embs = []
tgt_embs = []
for src_word, tgt_word in word_pairs.items():
    if src_word in src_embs_index and tgt_word in tgt_embs_index:
        src_embs.append(src_embs_index[src_word])
        tgt_embs.append(tgt_embs_index[tgt_word])

print('found ' + str(len(src_embs)) + ' words in the source language')
print('found ' + str(len(tgt_embs)) + ' words in the target language')

cossim_list = []
for i in tqdm(range(len(src_embs))):
    cossim_list.append(cosine_similarity( [src_embs[i]] , [tgt_embs[i]] )[0][0])
avg_cossim = sum(cossim_list) / len(cossim_list)
print('average cross-lingual cossim:', avg_cossim)