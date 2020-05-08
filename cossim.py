#!/usr/local/bin/python3

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

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

parser = argparse.ArgumentParser(description="calculate avg cossim between two sets of embeddings")
parser.add_argument("src_embs", help="specify source embeddings")
parser.add_argument("tgt_embs", help="specify target embeddings")
args = parser.parse_args()

trained_embs_index = load_embs_2_dict(args.src_embs)    # word:nparray
original_embs_index = load_embs_2_dict(args.tgt_embs)
# filtered_embs_index = {word: emb for word, emb in original_embs_index.items() if word in trained_embs_index}
filtered_embs_index = {word: original_embs_index[word] if word in original_embs_index else np.zeros(100) for word in trained_embs_index}
# some words found in Tokenizer are not present in the pre-trained BWE
# print(len(trained_embs_index))

cossim_list = []
for i in tqdm(range(len(trained_embs_index) - 1)):
    cossim_list.append(cosine_similarity( [list(trained_embs_index.values())[i]] , [list(filtered_embs_index.values())[i]] ))
# cossim_list = [cosine_similarity([emb1], [emb2]) for emb1 in trained_embs_index.values() for emb2 in filtered_embs_index.values()]
print(len(cossim_list))
avg_cossim = sum(cossim_list) / len(cossim_list)
print(avg_cossim)