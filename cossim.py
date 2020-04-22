#!/usr/local/bin/python3

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import utils

trained_embs_index = utils.load_embs_2_dict('./trained_embs.txt')    # word:nparray
original_embs_index = utils.load_embs_2_dict('EMBEDDINGS/EN_DE.txt.w2v')
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