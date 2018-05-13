import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
import pandas as pd
import re
import numpy as np


def make_set(df):
    all_words = []
    for _, row in df.iterrows():
        row['title'].replace("\'", "").replace("\"", "")
        row['content'].replace("\'", "").replace("\"", "")
        all_words.extend(re.findall(r'\w+|\S+', row['title']))
        all_words.extend(re.findall(r'\w+|\S+', row['content']))
    all_tokens = set(['<S>', '</S>'] + all_words)
    return all_tokens

def load_data():
    ALL_NEWS_DIR = 'all-the-news'
    df = pd.read_csv(ALL_NEWS_DIR+'/articles1.csv')
    # df = df.append(pd.read_csv(ALL_NEWS_DIR+'/articles2.csv'))
    # df = df.append(pd.read_csv(ALL_NEWS_DIR+'/articles3.csv'))
    df2 = df[['title','content']]
    return df2    

data = load_data()
all_tokens = make_set(data)
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
