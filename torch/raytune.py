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
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
#from functools import partial
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.schedulers import ASHAScheduler
import os



##########################
    
def load_data():
    datapath = "C:/Users/rober/Documents/projects/phishbot/torch/raydata"
    train_ids = torch.tensor(np.genfromtxt(datapath + '/train0.csv', delimiter=',').astype('int64'))
    val_ids = torch.tensor(np.genfromtxt(datapath + '/val0.csv', delimiter=',').astype('int64'))
    return train_ids, val_ids

# ray tune
def train(config, train_ids, df, test, translate):
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #dev = torch.device("cpu")
    #dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #writer = SummaryWriter()
    model = networks.NextNet(len(translate.vocab_dict), config['hidden_nodes'], config['hidden_layers'])
    model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_func = nn.CrossEntropyLoss() 

    #train_ids, val_ids = load_data()
    #data processing
    x, y = data_split(train_ids, sequence_length=config['sequence_length'])
    #xv, yv = data_split(val_ids, sequence_length=config['sequence_length'])

    ds = torch.utils.data.TensorDataset(x,y)
    #dsv = torch.utils.data.TensorDataset(xv, yv)

    train_dl = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
    #valid_dl = torch.utils.data.DataLoader(dsv, batch_size=config['batch_size'], shuffle=True)

    #del train_ids, val_ids, x, y, xv, yv, ds, dsv
    del train_ids, x, y, ds
    #trainset -> dataset, validation set -> datasetv

    for epoch in range(50):
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

        # if valid_dl != None:
        #     valid_loss = []
        #     model.eval()
        #     with torch.no_grad():
        #         for xb, yb in valid_dl:
        #             pred = model.forward(xb)
        #             for ii in range(len(yb)):
        #                 loss = loss_func(pred[ii], yb[ii])
        #                 valid_loss.append(loss)

        #     #print("validation", epoch, sum(valid_loss)/len(valid_loss))
        #     l = sum(valid_loss)/len(valid_loss)
        eval_feed = df[-1*config['sequence_length']:].copy()
        evals = evaluation.NextNetEvaluator(model, translate)
        winners=[]
        for ii in sorted(list(set(test.date))):
            temp = test[test.date == ii].Song
            #winners = []
            #for jj in range(5):
            newshow = evals.predict(eval_feed)
            _, number = evals.eval_preds(newshow, temp)
            winners.append(number)   
            evals.update_with_show(temp)

        tune.report(correct = sum(winners)/len(winners)) #average number of winners across 25 sets of predictions for the show
        

            #tune.report(loss=l.item())
    #print("Finished Tuning")


config = {
    "hidden_layers": tune.choice([1,2,3]),
    "hidden_nodes": tune.choice([256, 512, 1028]),
    #"learning_rate": tune.loguniform(1e-4, 1e-1),
    "learning_rate": tune.choice([.05,.06,.07,.08,.09]),
    "batch_size": tune.choice([8,16,32, 64, 128]),
    "sequence_length": tune.choice([100,150,200, 250, 300])
    #"data_dir": data_dir
}
# config = {
#     "hidden_layers": tune.choice([1]),
#     "hidden_nodes": tune.choice([256]),
#     #"learning_rate": tune.loguniform(1e-4, 1e-1),
#     "learning_rate": tune.choice([.025,.05,.07,.08,.09,.15]),
#     "batch_size": tune.choice([64]),
#     "sequence_length": tune.choice([100])
#     #"data_dir": data_dir
# }

#it works!
# analysis = tune.run(train, config=config, num_samples=2, resources_per_trial={"cpu": 8, "gpu": 1})

# #add optimizer
# hyperopt = HyperOptSearch(metric="correct", mode="max")
# analysis = tune.run(
#     train, 
#     config=config, 
#     num_samples=10, 
#     resources_per_trial={"cpu": 8, "gpu": 1},
#     search_alg = hyperopt,
#     metric = "loss",
#     mode = "min"
#     )

# #add early stopping
# hyperband = HyperBandScheduler()
# asha = ASHAScheduler(
#     max_t=20,
#     grace_period=1,
#     reduction_factor=2
#     )

 
#### lets setup actual performance instead of loss function to compare
data = pd.read_csv("C:/Users/rober/Documents/projects/phishbot/datacollection/final.csv")

#df = data[(data.date > "2009-01-01") & (data.date < "2019-01-01")].copy()
df = data[(data.date > "2009-01-01") & (data.date < "2019-01-01")].copy()
#df = data[data.date < "2019-01-01"].copy()
#test = data[data.date > "2019-02-20"]
test = data[(data.date > "2019-05-01") & (data.date < "2019-08-15")]
translate = Vocab(df.Song)
del data
#eval_feed = df[-sequence_length:]  #this needs to be just a series of songs

train_ids = translate.ids_from_songs(list(df.Song))  #this does not accept pandas series, needs to be a list

# def load_data():
#     datapath = "C:/Users/rober/Documents/projects/phishbot/torch/raydata"
#     train_ids = torch.tensor(np.genfromtxt(datapath + '/train0.csv', delimiter=',').astype('int64'))
#     val_ids = torch.tensor(np.genfromtxt(datapath + '/val0.csv', delimiter=',').astype('int64'))
#     return train_ids, val_ids
# train_ids,_ = load_data()

hyperopt = HyperOptSearch(metric="correct", mode="max")
#hyperband = HyperBandScheduler(metric="correct", mode="max")
#asha = ASHAScheduler(metric="correct", mode="max")
#medianboy = MedianStoppingRule(metric="correct", mode="max")

analysis = tune.run(
    tune.with_parameters(train, train_ids=train_ids, df=df, test=test, translate=translate),
    config=config, 
    num_samples=10, 
    resources_per_trial={"cpu": 8, "gpu": 1},
    search_alg = hyperopt,
    #scheduler=medianboy,
    stop={"training_iteration": 10, "correct": 4}
    # metric = "correct",
    # mode = "max"
    )



#analysis.get_best_config("correct", "max")
#analysis.dataframe().to_csv("raydata/3point0_rayresults.csv", index=False)