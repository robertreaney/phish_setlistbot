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



def load_data():
    datapath = "C:/Users/rober/Documents/projects/phishbot/torch/raydata"
    train_ids = torch.tensor(np.genfromtxt(datapath + '/train0.csv', delimiter=',').astype('int64'))
    val_ids = torch.tensor(np.genfromtxt(datapath + '/val0.csv', delimiter=',').astype('int64'))
    return train_ids, val_ids

train_ids, val_ids = load_data()

def train(config, train_ids, val_ids, epochs=10):
    #dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #writer = SummaryWriter()
    model = networks.NextNet(498, config['hidden_nodes'], config['hidden_layers'])
    model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_func = nn.CrossEntropyLoss() 

    #train_ids, val_ids = load_data()
    #data processing
    x, y = data_split(train_ids, sequence_length=config['sequence_length'])
    xv, yv = data_split(val_ids, sequence_length=config['sequence_length'])

    ds = torch.utils.data.TensorDataset(x,y)
    dsv = torch.utils.data.TensorDataset(xv, yv)

    train_dl = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
    valid_dl = torch.utils.data.DataLoader(dsv, batch_size=config['batch_size'], shuffle=True)

    del train_ids, val_ids, x, y, xv, yv, ds, dsv

    #trainset -> dataset, validation set -> datasetv
    final_val = []
    training_time = []
    for epoch in range(epochs):
        #training set
        model.train()
        tr_loss=[]
        start = time.time()
        for xb, yb in train_dl:
            pred = model(xb)
            for ii in range(len(yb)):
                loss = loss_func(pred[ii], yb[ii])
                if ii < len(yb):
                    loss.backward(retain_graph=True)
                if ii == len(yb):    
                    loss.backward(retain_graph=False)
            tr_loss.append(loss)

            opt.step()
                #opt.zero_grad()
            opt.zero_grad()
        #print(epoch, sum(tr_loss)/len(tr_loss))
        #writer.add_scalar('runs',sum(tr_loss)/len(tr_loss), epoch)
        #validation step   #this doesnt work anymore
        if valid_dl != None:
        
            valid_loss = []
            model.eval()
            with torch.no_grad():
                for xb, yb in valid_dl:
                    pred = model.forward(xb)
                    for ii in range(len(yb)):
                        loss = loss_func(pred[ii], yb[ii])
                        valid_loss.append(loss)
                    #valid_loss = sum(loss_func(model(xb.cuda()).view(len(xb),-1), yb.cuda()) for xb, yb in valid_dl)
                    #valid_loss = statistics.mean(loss_func(self.forward(xb), yb) for xb, yb in valid_dl)
        final_val.append((sum(valid_loss)/len(valid_loss)).item())
        training_time.append(start - time.time())
        average = sum(final_val[-10:])/len(final_val[-10:])
        print("validation", epoch, np.round(final_val[-1],2), np.round(average,2))
        if epoch > 10:
            if final_val[-1] - average > 0:
                break
    #return(loss=sum(valid_loss)/len(valid_loss))
    config['min'] = min(final_val)
    config['max'] = final_val[0]
    config['change'] = (config['max']-config['min'])/config['max']
    config['min_index'] = final_val.index(config['min'])
    config['time'] = sum(training_time)/len(training_time)
    config['epochs'] = epoch
    #print("Finished Tuning")
    return config
    


# config = {
#     "hidden_nodes": tune.choice([1,2]),
#     "hidden_layers": tune.sample_from(lambda _: 2**np.random.randint(8, 11)),
#     "learning_rate": tune.loguniform(1e-4, 1e-1),
#     "batch_size": tune.choice([2, 4, 8, 16]),
#     "sequence_length": tune.choice([10,25,50,100]),
#     #"data_dir": data_dir
# }

#train(config={'hidden_nodes': })

def gen_config():
    config = {
        "hidden_layers": int(np.random.choice([1,2,3])),
        "hidden_nodes": 2**int(np.random.choice([7,8,9,10,11])),
        "learning_rate": np.random.uniform(.08, .12, 1)[0],
        "batch_size": int(np.random.choice([5,10,20])),
        "sequence_length": int(np.random.choice([25,50,75,100,150,200])),
        #"data_dir": data_dir
    }
    print(config)
    return config

# train = pd.read_csv("raydata/train2.csv")
# val = pd.read_csv("raydata/val2.csv")
# translate = Vocab(train.Song)
# vocab_size = len(translate.vocab_dict) #this returns 510, but length of true vocab doesn't include ' ' or 'unknown


# np.savetxt("raydata/val0.csv", val_ids.numpy(), delimiter=",")
# np.savetxt("raydata/train0.csv", train_ids.numpy(), delimiter=",")
# #test one training
#res = train(gen_config())

results = []

for ii in range(50):
    results.append(train(gen_config(),train_ids, val_ids, epochs=100))
    print(ii)


#### collect many dicts with the same keys
d = {}
for k in results[0].keys():
  d[k] = tuple(d[k] for d in results)

d

import pandas as pd
df = pd.DataFrame(d)
df.to_csv("tuning/secondbatch.csv", index=False)


import matplotlib.pyplot as plt
plt.hist(df[df['min'] < 8.5]['sequence_length'])
plt.show()