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
#i have seen two approaches
#   1) give each song an integer representation and a setlist is now a list of numbers
#   2) create indicator column for each song, and setlist is a matrix. row =song, column = song indicator
# number 1 sounds less awful for now. later i will see if 2 does better
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.device()
data = pd.read_csv("final_nosoundchecks_3poin0.csv")
data

train = data[data.date < "2019-06-11"]
val = data[(data.date >= "2019-06-11") & (data.date < "2019-07-01")]
val
test = data[(data.date >= "2019-07-01") & (data.date < "2019-08-28")]

test = test.Song
test=list(test)
val = val.Song
val = list(val)
train = train.Song
train = list(train) #now data is a list of songs
len(train) #now we have list of just songs


class Vocabulary():
    """
    This class handles the translation between unique strings and numerical IDs. It assigns an ID and then provides functions
    for translating Song <=> ID. Initialize it by giving it all of your training songs in a pandas.series or list.
    """
    def __init__(self, songs):
        """
        Args: training songs as pandas.series, numpy.array, or list
        """
        #make the vocab input have no repeats if user messed up
        songs = sorted(set(songs)) #make sure no repeats. sort just for better reference
        #make vocab dict
        self.vocab_dict = {song: (i+1) for i, song in enumerate(songs)}
        #make id dict
        self.id_dict = dict()
        for song in self.vocab_dict:
            self.id_dict[self.vocab_dict[song]] = song

    def ids_from_songs(self, data):
        """
        Function to retreive ids given songs.
            Args: song or list of songs
        """
        #make sure input value is good
        want_list = True
        #If user inputs a string rather than list. we want to return integer rather than [int]. For conveinence
        #also allows user to do something like ids_from_songs("Fluffhead") and not get an error
        if(type(data) != list): 
            want_list = False
            data = [data]
        id_series = []
        for ii in data:
            try:
                id_series.append(self.vocab_dict[ii])
            except:
                id_series.append(0) #in case item is not in our dictionary, return 0
        if(want_list):
            return id_series
        else:
            return id_series[0]

    #as named, retreives songs from ids
    def songs_from_ids(self, data):
        """
        Function to retreive songs given ids.
            Args: ID or list of IDs
        """
        want_list = True
        if(type(data) != list): 
            want_list = False
            data = [data]
        #let user input an integer as well just to avoid possible errors later
        #if(type(data) != list):
        song_series = []
        for ii in data:
            #assert ii in self.id_dict, "Song ID: %s not recognized. Valid IDs are in self.id_dict"%ii
            song_series.append(self.id_dict[ii])
            # try:
            #     song_series.append(self.id_dict[ii])
            # except:
            #     song_series.append("Unknown")
        if want_list:
            return song_series
        else:
            return song_series[0]

translate = Vocabulary(np.asarray(train))

########
#Dataset construction
# ids_dataset = translate.ids_from_songs(train)
# ids_dataset
# #1) split up data into sequences of len=sequence_length+1
#2) split sequences to (input,label)


def input_label_split(data, sequence_length, overlap_inputs=True):
    """
    Breaks data up from a long list to (input, label) pairs where len(input) = sequence_length
    """
    x = []
    y = []
    #output data is (input, output) = (x,y)
    #unlike tf we will do this
    #[1,2,3,4] with seq_len=3 => [1,2,3], [2,3,4]. aka we will overlap input label sequences. maybe this
    if overlap_inputs:
        num_sequences = len(data) - sequence_length #buffer required to make first sequence
        assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
        for ii in range(num_sequences):
            x.append(list(data[0+ii:sequence_length+ii]))
            y.append(data[sequence_length+ii])
    else: #make it like tf where. [1,2,3,4] => [1,2,3] 
        num_sequences = len(data) // (sequence_length + 1)
        assert num_sequences > 0, "You created zero data entries. Use more data or a smaller sequence_length"
        for ii in range(num_sequences):
            shift = sequence_length + 1
            x.append(list(data[shift*ii : sequence_length+(shift*ii)]))
            y.append(data[sequence_length+(ii*shift)])
    #make into a TensorDataset
    from torch.utils.data import TensorDataset
    #dataset = TensorDataset(torch.tensor(x, device=dev), torch.tensor(y, device=dev))
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y)) #not sure if putting these on GPU is actually better or not
    return dataset

train_ids = translate.ids_from_songs(train)
train_ds = input_label_split(train_ids, 100, overlap_inputs=True)
#works with song names or song its
val_ids = translate.ids_from_songs(val)
val_ds = input_label_split(val_ids, 10, overlap_inputs=True)

test_ids = translate.ids_from_songs(test)
test_ds = input_label_split(test_ids, 10, overlap_inputs=True)
#now we need to batch this with dataloader
batch_size=64
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

#embedding_dim = 1000
#sequence_length = 200


class SetlistNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, sequence_length=100, hidden_size=500, num_layers=2, reccur_type="GRU"):
        super(SetlistNet, self).__init__()
        #some constants
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.reccur_type = reccur_type
        #network structure
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if reccur_type=="GRU":
            self.gru = nn.GRU(input_size=(embedding_dim*sequence_length), hidden_size=hidden_size)#, hidden_size=gru_hs, num_layers=gru_layers) #might have to use these arguements, not sure
        else:
            assert reccur_type=="LSTM", "Please enter LSTM or GRU for reccurence layer type"
            #self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.lstm = nn.LSTM(embedding_dim*sequence_length, hidden_size)        
        self.dense = nn.Linear(hidden_size, vocab_size)
        #self.label = nn.Softmax(dim=1)
    def forward(self, inputs, states=None, return_states=False):
        x = self.embeddings(inputs)#.view(len(inputs), 1,-1) #im not sure why i dont or do need this
        #x = self.embeddings(inputs).flatten() #creates size len(inputs)*embedding_dim output
        if self.reccur_type == "GRU":
            x, states = self.gru(x.view(len(inputs),1,-1))
        else:
            x, states = self.lstm(x.view(len(inputs),1,-1))  
        x = self.dense(x)
        #x = self.label(x.view(1,-1))
        if return_states:
            return x, states
        else:
            return x


len(set(train))
lr = .001
model = SetlistNet(vocab_size=len(set(train))+1, reccur_type="LSTM")
model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = F.cross_entropy
loss_f = torch.nn.CrossEntropyLoss(reduction='mean')
#loss_f = nn.NLLLoss()

def fit(model, opt, loss_func, train_dl, valid_dl=None, epochs=2):
    #dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    writer = SummaryWriter()
    for epoch in range(epochs):
        #training set
        model.train()
        tr_loss=[]
        for xb, yb in train_dl:
            pred = model(xb.cuda())
            loss = loss_func(pred.view(len(xb), -1), yb.cuda())
            tr_loss.append(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(epoch, sum(tr_loss)/len(tr_loss))
        #writer.add_scalar('runs',sum(tr_loss)/len(tr_loss), epoch)
        #validation step
        if valid_dl != None:
            model.eval()
            with torch.no_grad():
                valid_loss = sum(loss_func(model(xb.cuda()).view(len(xb),-1), yb.cuda()) for xb, yb in valid_dl)
            print(epoch, valid_loss/len(valid_dl))


#torch.cuda.set_device(dev)
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
writer = SummaryWriter()



fit(model, opt, loss_f, train_dl, epochs=5)

loss_f(model(xx.cuda()).view(len(xx), -1), yy.cuda()) #6.3386, #2.5285
loss_func(model(xx.cuda()).view(len(xx),-1), yy.cuda()) #6.1907
sum(loss_f(model(xb.cuda()).view(len(xb),-1), yb.cuda()) for xb,yb in test_dl)/len(test_dl)
xx.cuda()



max(train_ids)
min(train_ids)
min(val_ids)
max(val_ids)

xx,yy = next(iter(train_dl))

out = model(xx.cuda())
out
loss_func(out.view(len(xx),-1), yy.cuda())
loss_f(out.view(len(xx),-1), yy.cuda())


#make input sequences for prediction



eval_feed = train[-100:]
ids = translate.ids_from_songs(eval_feed)
newshow = []
states = None
for ii in range(25):
    tens = torch.tensor([ids])

    model.eval()
    with torch.no_grad():
        probs, states = model(tens.cuda(),states, return_states=True)
        prob = probs.detach().cpu().numpy()
        prob = np.exp(prob) / (np.exp(prob)).sum()
        #best_prob = np.argmax(prob)
        random_prob = np.random.choice(np.arange(1,497),p=prob[0][0])
        #next_song = translate.songs_from_ids(best_prob)
        next_song = translate.songs_from_ids(random_prob)
        newshow.append(next_song)
    ids = ids[:-1]
    ids.append(translate.ids_from_songs(next_song))


newshow
targets = data[data.date == "2019-06-11"]
targets = np.asarray(targets.Song)
winners = []
count = 0
for ii in newshow:
    if ii in targets:
        count += 1
        winners.append(ii)

winners
count
targets
newshow

