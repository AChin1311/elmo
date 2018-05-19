import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings
import numpy as np
import pickle

def make_set(data):
    all_words = []
    for line in data:
        all_words.extend(' '.join(line).split())
    all_tokens = set(['<S>', '</S>'] + all_words)
    return all_tokens

def pickle2tuple(nr):
        [heads, desc, _] = pickle.load(open('pickles/all-the-news_'+nr+'.pickle', 'rb'))
        return heads + desc

def tokenized(data):
    tokenized_sentences = []
    for article in data:
        for line in article:
            tokenized_sentences.append(line.split())
    return tokenized_sentences

data = pickle2tuple('5000')
all_tokens = make_set(data)
tokenized_sentences = tokenized(data)

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

tf.reset_default_graph()


## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)


with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_sentences)
    
    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_ = sess.run(
        [elmo_context_input['weighted_op']],
        feed_dict={context_token_ids: context_ids}
    )
    print(elmo_context_input_[0])



