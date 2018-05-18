import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
import numpy as np
import pickle

def make_set(data):
    all_words = []
    for line in data:
        print(line)
        all_words.extend(' '.join(line).split())
    all_tokens = set(['<S>', '</S>'] + all_words)
    return all_tokens

def pickle2tuple(nr):
        [heads, desc, _] = pickle.load(open('pickles/all-the-news_'+nr+'.pickle', 'rb'))
        print(len(heads))
        print(len(desc))
        print(type(heads), type(desc))
        return heads + desc

data = pickle2tuple('5000')
print(len(data))
all_tokens = make_set(data)
print(len(all_tokens))

vocab_file = 'vocab_all_news.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

# Location of pretrained LM.  Here we use the test fixtures.
options_file = './options.json'
weight_file = './weights.hdf5'
# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = 'elmo_token_embeddings.hdf5'
tokens, embeddings = dump_token_embeddings(vocab_file, options_file, weight_file, token_embedding_file)

np.save('elmo_embeddings', embeddings)
np.save('tokens', tokens)

# embeddings = np.load('elmo_embeddings.npy')
# tokens = np.load('tokens.npy')

with open("elmo_all_news.txt", 'w') as f:
    for i in range(len(tokens)):
        f.write(tokens[i].decode('utf-8')+' ')
        f.write(' '.join(["%.8f" % e for e in embeddings[i, :]]))
        f.write('\n')     
