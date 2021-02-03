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

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

import os



##########################
# ray tune
def train(config, epochs=10):
    #dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #writer = SummaryWriter()
    model = networks.NextNet(5, config['hidden_nodes'], config['hidden_layers'])
    model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_f = nn.CrossEntropyLoss() 

    train_ids, val_ids = load_data()
    #data processing
    x, y = data_split(train_ids, sequence_length=config['sequence_length'])
    xv, yv = data_split(val_ids, sequence_length=config['sequence_length'])

    ds = torch.utils.data.TensorDataset(x,y)
    dsv = torch.utils.data.TensorDataset(xv, yv)

    train_dl = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
    valid_dl = torch.utils.data.DataLoader(dsv, batch_size=config['batch_size'], shuffle=True)

    del train_ids, val_ids, x, y, xv, yv, ds, dsv

    #trainset -> dataset, validation set -> datasetv

    for epoch in range(epochs):
        #training set
        model.train()
        tr_loss=[]
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
            #print("validation", epoch, sum(valid_loss)/len(valid_loss))
        tune.report(loss=sum(valid_loss)/len(valid_loss))
    print("Finished Tuning")




def ray_execute(num_samples=10, max_epochs=10):
    #data_dir = os.path.abspath("./raydata")
    data_dir = "C:/Users/rober/Documents/projects/phishbot/torch/raydata"
    load_data()

    config = {
    "hidden_layers": tune.choice([1,2]),
    "hidden_nodes": tune.sample_from(lambda _: 2**np.random.randint(2, 5)),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16]),
    "sequence_length": tune.choice([10,25,50]),
    #"data_dir": data_dir
}

    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
        )
    result = tune.run(
        tune.with_parameters(train),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        #progress_reporter=reporter,
        checkpoint_at_end=False,
        metric="loss",
        mode="min"
        )
    best_trial = result.get_best_trail("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"])) 
    
def load_data():
    datapath = "C:/Users/rober/Documents/projects/phishbot/torch/raydata"
    train_ids = torch.tensor(np.genfromtxt(datapath + '/train.csv', delimiter=',').astype('int64'))
    val_ids = torch.tensor(np.genfromtxt(datapath + '/val.csv', delimiter=',').astype('int64'))
    return train_ids, val_ids

ray_execute()
2**11