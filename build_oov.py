import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm

dataset1 = 'question1_sub_bal'
dataset2 = 'question2_sub_bal'

doc_content_list1 = []

with open('data/corpus/' + dataset1 + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list1.append(line.strip())

doc_content_list2 = []

with open('data/corpus/' + dataset2 + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list2.append(line.strip())

# build corpus vocabulary
word_set = set()

for doc_words in doc_content_list1:
    words = doc_words.split()
    word_set.update(words)

for doc_words in doc_content_list2:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)
vocab_size = len(vocab)

oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

with open("data/oov_dic", 'wb') as f:
    pkl.dump(oov, f)