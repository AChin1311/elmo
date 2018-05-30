from __future__ import print_function
from allennlp.modules.elmo import _ElmoBiLm
import torch.nn as nn
import torch
from torch import FloatTensor,LongTensor,ByteTensor, Tensor
import torch.nn.functional as f
from collections import OrderedDict
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from torch.autograd import Variable
import nltk,os
import numpy as np
import argparse
import pickle
from allennlp.modules.elmo import Elmo
#from nn_layer import EmbeddingLayer, Encoder


def get_sentences(nr="50"):
    print("loading ", 'pickles/all-the-news_'+nr+'.pickle')
    [heads, desc, _] = pickle.load(open('pickles/all-the-news_'+nr+'.pickle', 'rb'))
    print(len(heads), " news loaded!")
    
    articles = []
    for h, d in zip(heads, desc):
        art = ""
        art += h[0]
        for l in d:
            art += l
        articles.append(art)
    print(len(articles))
    return articles

def elmo_sent_mapper(sentence, max_length, pad_token="~"):
    word_list = []
    for i in range(max_length):
        word = sentence[i] if i < len(sentence) else pad_token
        word_list.append(ELMoCharacterMapper.convert_word_to_char_ids(word))
    return word_list

def get_batches(data, batch_size):
    batched_data = []
    for i in range(len(data)):
        if i % batch_size == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    return batched_data

def batch_sentence_mapper(batch, maxl):
    return Variable(LongTensor([elmo_sent_mapper(sent,maxl) for sent in batch]))
    
def store_batch_embeddings(sl,emb_red,num_rec,batch_size, max_sent_len):
    num_sent = len(sl)
    emb_red = emb_red.data.numpy()
    count = 0
    #print(sum([len(s) for s in sl] ))
    with open(embedding_file,'a+') as fil:
        for i in range(num_sent):
            for j in range(len(sl[i])):
                word_embedding = emb_red[i][j]
                fil.write('{0} {1}\n'.format(sl[i][j],' '.join(map(str,word_embedding))))
                count+=1
    return count
    
def get_elmo_embeddings(sl, num_rec, batch_size, dim):
    if os.path.exists(embedding_file):
        print(embedding_file," already exists. Do you still want to proceed?")
        x = input("y/n")
        if x=='y':
            print("Continuing..")
        else:
            return
    elmo_embedder = Elmo('options.json', 'weights.hdf5',num_output_representations=1, requires_grad=False)
    if use_cuda and do_dot_cuda:
        print("\tRunning elmo.cuda")
        elmo_embedder.cuda()
    batched_data = get_batches(sl, batch_size)
    print("\t{0} sentences in {1} records and generated {2} batches each of {3} sentences".format(len(sl),num_rec,len(batched_data),batch_size))
    bno = 0
    wc  = 0
    for batch in batched_data:
        max_sent_len = max([len(s) for s in batch])
        mapped_sentences = batch_sentence_mapper(batch, max_sent_len)
        act = elmo_embedder(mapped_sentences)['elmo_representations']
        emb_red = act[0]
        if use_cuda and do_dot_cuda:
            emb_red = emb_red.cpu()
        cnt = store_batch_embeddings(batch, emb_red,num_rec,batch_size, max_sent_len)
        bno+=1
        wc +=cnt
        print("\t\tStored batch {0} [with {1} words]".format(bno,cnt))
    print("Generated embeddings for data with {0} words".format(wc)) 
    return

if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  if use_cuda:
      print("Cuda available")

  num_records = 10000
  batch_size = 256
  target_d =300
  do_dot_cuda = 0

  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--n_rec",type=int, default = 10000,help="Number of records")
  parser.add_argument("-d", "--dimension", type=int, default=300,help= "Target dimension of embeddings")
  parser.add_argument("-b", "--batchsize", type=int, default=100,
                      help="batch size of sentences to store")
  parser.add_argument("-i", "--do_dot_cuda", type=int, default=0,
                      help="Target dimension of embeddings")
  parser.add_argument('--gpu', type=int, default=0, help='gpu id')
  args = parser.parse_args()
  ####

  if use_cuda:
    if args.gpu >= 0:
      print("Setting cuda device")
      torch.cuda.set_device(args.gpu)
      print("Cuda set!")
      torch.set_default_tensor_type('torch.cuda.FloatTensor')

  target_d = args.dimension
  num_records = args.n_rec
  batch_size = args.batchsize
  do_dot_cuda = args.do_dot_cuda
  print(" *********** Generating {0} dimension embeddings for {1} records with batch size {2} *************".format(target_d,num_records,batch_size))

  embedding_file = 'new-elmo_embed_nr-{0}_dim-{1}_bsiz-{2}.txt'.format(num_records,target_d,batch_size)

  ##GPU code
  FloatTensor = torch.cuda.FloatTensor if use_cuda and do_dot_cuda else torch.FloatTensor
  LongTensor = torch.cuda.LongTensor if use_cuda and do_dot_cuda else torch.LongTensor
  ByteTensor = torch.cuda.ByteTensor if use_cuda and do_dot_cuda else torch.ByteTensor

  DIR = './pickles'
  nltk.download('punkt')
  english_sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

  print('Getting sentence lists')

  sent_list=get_sentences()
  split_sent_list = [[w.lower() for w in s.split()] for s in sent_list]

  print('Splitting sentence lists and converting to lower case words')

  lengths=[len(s) for s in split_sent_list]
  # type(lengths)
  wc = sum(lengths)
  print("\n\nNow, Generating embeddings for data with {0} words".format(wc))
  #plt.hist(lengths, bins=np.arange(min(lengths), max(lengths)+1))
  #plt.plot()

  get_elmo_embeddings(split_sent_list, num_records, batch_size, target_d)
