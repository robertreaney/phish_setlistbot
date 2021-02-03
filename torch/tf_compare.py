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
import statistics

from processing import input_label_split, data_split
from vocabulary import Vocab
import networks
import evaluation


import timeit



######################
#embedding_dim = 50
# hidden_nodes = 1000
# hidden_layers = 1
# sequence_length = 100
# batch_size = 20
# learning_rate = .1



# compare outputs to tf

#data import
data = pd.read_csv("C:/Users/rober/Documents/projects/phishbot/datacollection/final.csv")

test = []
test.append(data[data.date == "2019-07-12"].Song)
test.append(data[data.date == "2019-07-13"].Song)
test.append(data[data.date == "2019-07-14"].Song)
test.append(data[data.date == "2019-08-30"].Song)
test.append(data[data.date == "2019-08-31"].Song)
test.append(data[data.date == "2019-09-01"].Song)
val = data[data.date > "2019-07-11"].Song

data3point0 = data[(data.date > "2009-01-01") & (data.date < "2019-07-12")]

df = data3point0['Song'].copy()
del data, data3point0

#vocab
translate = Vocab(df)
# translate.ids_from_songs(["eaefe"])
# translate.ids_from_songs(['Possum', 'Sigma Oasis'])
# translate.ids_from_songs("Fluffhead")
# translate.songs_from_ids([0,1,2])
# translate.songs_from_ids([111111])
vocab_size = len(translate.vocab_dict) #this returns 510, but length of true vocab doesn't include ' ' or 'unknown
# df transformation
train_ids = translate.ids_from_songs(list(df))  #this does not accept pandas series, needs to be a list
len(train_ids)
val_ids = translate.ids_from_songs(list(val))

val_ids
#all_ids
# np.savetxt("raydata/val.csv", val_ids.numpy(), delimiter=",")
# np.savetxt("raydata/train.csv", train_ids.numpy(), delimiter=",")

#x, y = input_label_split(train_ids, seq_length=sequence_length, overlap_inputs=False)
x, y = data_split(train_ids, sequence_length=sequence_length)
xv, yv = data_split(val_ids, sequence_length=sequence_length)

ds = torch.utils.data.TensorDataset(x,y)
dataset = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

dsv = torch.utils.data.TensorDataset(xv, yv)
datasetv = torch.utils.data.DataLoader(dsv, batch_size=batch_size, shuffle=True)


#xb, yb = next(iter(dataset))  #(batchsize x sequence_length), batchsize
#xb.shape
model = networks.NextNet(vocab_size, hidden_nodes, hidden_layers)
model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_f = nn.CrossEntropyLoss()

#predsb = model(xb) #LOGITS

#######

model.fit(opt, loss_f, dataset, datasetv, epochs=10)


###### make predictions
#1) make predictions for next show
#2) update inputs with new show
#3) repeat
eval_feed = df[-sequence_length:] 
evals = evaluation.NextNetEvaluator(model, translate)

winners = []
for ii in range(25):
    newshow = evals.predict(eval_feed)
    succeses, number = evals.eval_preds(newshow, test[0])
    winners.append(number)
statistics.mean(winners)


