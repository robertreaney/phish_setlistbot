
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import pandas as pd
import torchtext
import numpy as np
from torch.utils.data import TensorDataset


# len(translate.vocab_dict)**.25
# n_vocab = len(translate.vocab_dict)
# n_embed = math.ceil(n_vocab**.25)
# emb = nn.Embedding(n_vocab, n_embed)
# emb.cuda()
# emb(xb).shape #(batch_size, sequence_length, n_embedd)
# emb(xb).shape

#             #input of shape (seq_length, batch, input_size)
# gru = nn.GRU(input_size=(embedding_dim*sequence_length), hidden_size=hidden_size)#, hidden_size=gru_hs, num_layers=gru_layers) #might have to use these arguements, not sure
# gru = nn.GRU(5, 20, 2)
# gru.cuda()
# lstm = nn.LSTM(5, 20, 2) #(inputsize, hidden nodes, hidden layers)
# lstm.cuda()
# inn, (hn, cn) = lstm(emb(xb))
# inn.shape #batchsize, sequence_length, hiddennode
# innn, hn = gru(emb(xb))
# innn.shape #batchsize, sequence_length, hiddennode

# linear = nn.Linear(20, n_vocab)
# linear.cuda()
# linear(inn).shape   #this has the correct shape

class NextNet(nn.Module):
    def __init__(self, vocab_size, hidden_nodes, hidden_layers, reccur_type="GRU"):
        super(NextNet, self).__init__()
        assert reccur_type in ["GRU", "LSTM"], "Only GRU or LSTM recurrance supported."
        self.reccur_type = reccur_type
        n_embed = math.ceil(vocab_size**.25)
        #network structure
                                    
        self.embeddings = nn.Embedding(vocab_size, n_embed) #output.shape = (batchsize, seq_length, n_embed)
                #(inputsize, hidden nodes, hidden layers)
        self.gru = nn.GRU(n_embed, hidden_nodes, hidden_layers)
        self.lstm = nn.LSTM(n_embed, hidden_nodes, hidden_layers)        
        self.dense = nn.Linear(hidden_nodes, vocab_size)
        #self.softmax = nn.Softmax()

    def forward(self, inputs, states=None, return_states=False):
        x = self.embeddings(inputs)
        if self.reccur_type == "GRU":
            if states == None:
                x, states = self.gru(x)
            else:
                x, states = self.gru(x, states)
        if self.reccur_type == 'LSTM':
            if states == None:
                x, states = self.lstm(x) 
            else:
                x, states = self.lstm(x, states)
        x = self.dense(x)
        if return_states:
            return x, states
        else:
            return x
