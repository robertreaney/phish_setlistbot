#import tensorflow as tf
import pandas as pd
import numpy as np
#from tensorflow.keras.layers.experimental import preprocessing
import math
import os
import time
from tqdm import tqdm
import torch

data = pd.read_csv("datacollection/final.csv")
data
df = data[['Set', 'Song', 'show']]
del data

#######
temp = data[data.show == 3]
'Set 2' not in np.asarray(temp.Set)
np.asarray(temp.date)[0]
bad_shows[:5]

[ii for ii in sorted(list(set(data.date))) if ii not in bad_shows][:10]
data[data.date == '1994-04-23']

def remove_useless_shows(data):
    bad_shows = []
    #gotta remove shows without two sets as a naive rule to exclude soundchecks
    for ii in list(set(data.show)):
        temp = data[data.show==ii]
        if 'Set 2' not in np.asarray(temp.Set):
            bad_shows.append(np.asarray(temp.date)[0])
    return data[data.date.isin([ii for ii in sorted(list(set(data.date))) if ii not in bad_shows])]

df = remove_useless_shows(data)
#df.to_csv("final_nosoundchecks.csv", index=False)



df = pd.read_csv("final_nosoundchecks.csv")


def make_openclose_df(df):#THIS PROBABLY NEEDS TO BE REDONE TO OCCUR AFTER TOKENIZATION SO WE CAN FORCE EMPTY SLOTS TO BE INTEGER 0 and not string 0
    """
    This is the data processing for predicting [set1o, set1c, set2o, set2c]
    Input is the setlist (padded to length 52) and output is length 4 list
    """#THIS PROBABLY NEEDS TO BE REDONE TO OCCUR AFTER TOKENIZATION SO WE CAN FORCE EMPTY SLOTS TO BE INTEGER 0 AFTERWARD

    max_songs = df.groupby('date').count().max()[0]
    for ii in tqdm(list(set(df.show))):
        #print(ii)
        temp = df[df.show == ii]
        newinputs = list(np.zeros(max_songs))  
        currentsongs = np.asarray(temp.Song)
        newinputs[0:len(currentsongs)] = currentsongs 
        newoutput = np.asarray([np.asarray(temp.Song)[0], np.asarray(temp[temp.Set == 'Set 1'].Song)[-1], np.asarray(temp[temp.Set == 'Set 2'].Song)[0], np.asarray(temp[temp.Set == 'Set 2'].Song)[-1]])
        try:
            inputs = np.vstack((inputs, newinputs))
            outputs = np.vstack((outputs, newoutput))
        except:
            inputs = newinputs
            outputs = newoutput
    return inputs, outputs


inputs, outputs = make_openclose_df(df) 
make_openclose_df(df[df.show==1000]) 


assert "Set 1" == "Set 1"
########################
#do something similar but have (setlist, y) row for every set1/set2/encore song appearance
#also should probably be done with tensors to be faster
def make_wildcard_df(df, set_choice): 
    #THIS PROBABLY NEEDS TO BE REDONE TO OCCUR AFTER TOKENIZATION SO WE CAN FORCE EMPTY SLOTS TO BE INTEGER 0 and not string 0
    """ choose set = ["Set 1", "Set 2", "Encore"] and receive input, output pairs in (setlist, song) format for desired set
        function will ouput inputs, outputs... a dataframe for your desired set
        """
    assert set_choice in ["Set 1", "Set 2", "Encore"], "set_choice must be from ['Set 1', 'Set 2', 'Encore']"
    #i dont want to get ride of data for all sets. i need them for the input
    #df = df[df.Set == set_choice] #get rid of all rows not of desired set
    max_songs = df.groupby('date').count().max()[0] #find out what the longest show in your df is to set max size for input padding
    for ii in tqdm(list(set(df.show))): #for each show      
        temp = df[df.show == ii] #data for a certain show only
        #songs in desired set of that show
        newinputs = list(np.zeros(max_songs))  #this is the setlist of the current show
        currentsongs = np.asarray(temp.Song)
        newinputs[0:len(currentsongs)] = currentsongs 
        for song in list(temp[temp.Set == set_choice].Song): #for every song in the desired set of a show make a row with this as the output
            try:
                inputs = np.vstack((inputs, newinputs))
                outputs.append(song)
            except:
                inputs = newinputs
                outputs = [song]
    return inputs, outputs



inputs, outputs = make_wildcard_df(df, "Set 1") #takes a while for set1 and set2
inputs, outputs = make_wildcard_df(df[df.show==10], "Set 1")





















####old tf stuff, not sure if needed anymore
dataset = tf.data.Dataset.from_tensor_slices((input, output))
#########
# structure data for set1/set2 wildcard NN
####hey this is how you structure things for this Dataset.from_tensor_slices() function
temp = tf.data.Dataset.from_tensor_slices((np.asarray([np.asarray([1,2,3]),np.asarray([4,5,7])]), np.asarray([1,2])))


for feat, targ in temp.take(2):
  print ('Features: {}, Target: {}'.format(feat, targ))
#######
df = data[data.date > "2009-01-01"]
def split_date_wildcard(df, input_length):
    input_length = 100
    datebuffer = df.iloc[100].date
    #target_date = df[df.date > datebuffer].iloc[0].date
    df_set1, df_set2, df_encore = 0, 0, 0
    for target_date in sorted(list(set(df[df.date > datebuffer].date))):
        target_firstindex = df[df.date == target_date].index[0]
        #inputs
        previous_songs = np.asarray(df[df.index.isin(np.arange(target_firstindex-100, target_firstindex))].Song)###here are the previous 100 songs before the show of interest
        #outputs
        set1 = np.asarray(df[(df.date == target_date)&(df.Set == 'Set 1')].Song)
        set2 = np.asarray(df[(df.date == target_date)&(df.Set == 'Set 2')].Song)
        encore = np.asarray(df[(df.date == target_date)&(df.Set == 'Encore')].Song)

        for ii, jj in zip([df_set1, df_set2, df_encore], [set1, set2, encore]):
            for kk, item in enumerate(jj): #make a row for each label
                if ii == 0:
                    ii = [previous_songs, jj[kk]]
                else:
                    ii[0] = np.vstack((ii[0], previous_songs))
                    ii[1] = np.vstack((ii[1], np.asarray(jj[kk])))

    for ii in [df_set1, df_set2, df_encore]:
        ii[1] = ii[1].flatten()

    return df_set1, df_set2, df_encore
##### this isn't quite working
df1, df2, df3 = split_date_wildcard(df, 100)


    return df_set1, df_set2, df_encore




########
class VocabProcessor():
    def __init__(self,songs):
        self.vocab = sorted(set(songs))
        #self.ids_from_songs = preprocessing.StringLookup(vocabulary=list(sorted(set(songs))))
        self.ids_from_songs = preprocessing.StringLookup(vocabulary=list(self.vocab))
        self.songs_from_ids = preprocessing.StringLookup(vocabulary=self.ids_from_songs.get_vocabulary(), invert=True)
        #self.songs_from_ids = preprocessing.StringLookup(vocabulary=preprocessing.StringLookup(vocabulary=list(sorted(set(songs)))).get_vocabulary(), invert=True)
        self.songs = songs
    # def get_dataset(self):
    #     #self.all_ids = self.ids_from_songs(self.songs) #converts every song played to numeric vocab
    #     #makes dataset of the 1d tensor and makes them individual objects
    #     #self.ids_dataset = tf.data.Dataset.from_tensor_slices(self.all_ids)
    #     #return self.ids_dataset
    #     return tf.data.Dataset.from_tensor_slices(self.ids_from_songs(self.songs))
    def compact_songbytes(self, songs):
        return tf.compat.as_str_any(songs.numpy())



process = VocabProcessor(df.Song)


for ii in ids_data.take(10):
    #print(process.songs_from_ids(ii))
    print(process.compact_songbytes(process.songs_from_ids(ii)))
    


######
# input/output preparation
######


#outputs -> openers, closers, wildcard, encore

#need to split dataset into (input, output) pairs FOR TRAINING
#these functions all require a "Song" and "Set" column
def split_input_target(show):
    input_songs = np.asarray(show.Song)
    #set 1 opener, set1 closer, set2 open, set2 close, encore
    opener = input_songs[0]
    set1closer = np.asarray(show[show.Set == 'Set 1'].Song)[-1]
    set2opener = np.asarray(show[show.Set == 'Set 2'].Song)[0]
    set2closer = np.asarray(show[show.Set == 'Set 2'].Song)[-1]
    target_songs = [opener, set1closer, set2opener, set2closer]
    return input_songs, target_songs

def split_set1(show):
    input_songs = np.asarray(show.Song)
    set1 = np.asarray(show[show.Set == 'Set 1'].Song)
    return input_songs, set1

def split_set2(show):
    input_songs = np.asarray(show.Song)
    set2 = np.asarray(show[show.Set == 'Set 2'].Song)
    return input_songs, set2

def split_encore(show):
    input_songs = np.asarray(show.Song)
    encore = np.asarray(show[show.Set == 'Encore'].Song)
    return input_songs, encore

# #this wants only a sequence of songs
# def split_input_target(sequence):
#     input_songs = sequence[:-1]
#     target_songs = sequence[1:]
#     return input_songs, target_songs


#later we will calculate gap



#(set1open, set1close, set2open, set2close)
input_songs, target_songs = split_input_target(show)
input_songs
target_songs

split_set1(show) #set1 wildcard
split_set2(show) #set2 wildcard
split_encore(show) #encore


class DataProcessor():
    def __init__(self,df):
        self.vocab = sorted(set(df.Song))
        #self.ids_from_songs = preprocessing.StringLookup(vocabulary=list(sorted(set(songs))))
        self.ids_from_songs = preprocessing.StringLookup(vocabulary=list(self.vocab))
        self.songbytes_from_ids = preprocessing.StringLookup(vocabulary=self.ids_from_songs.get_vocabulary(), invert=True)
        #self.songs_from_ids = preprocessing.StringLookup(vocabulary=preprocessing.StringLookup(vocabulary=list(sorted(set(songs)))).get_vocabulary(), invert=True)
        self.songs = df.Song
        self.show = df.show
        self.set = df.Set
    def get_id_dataset(self):
        #self.all_ids = self.ids_from_songs(self.songs) #converts every song played to numeric vocab
        #makes dataset of the 1d tensor and makes them individual objects
        return tf.data.Dataset.from_tensor_slices(self.ids_from_songs(self.songs))
    def songs_from_ids(self, song_ids):
        return tf.compat.as_str_any(self.songbytes_from_ids(song_ids).numpy())
    def show_split(self):


process = DataProcessor(df)
ids_dataset = process.get_id_dataset()



for ii in process.get_id_dataset().take(10):
    print(process.songs_from_ids(ii))
    #print(process.compact_songbytes(process.songs_from_ids(ii)))

tf.constant([1,2,3])

data.groupby(["show"]).count().max()
#### MAX SETLIST LENGTH IS 51 FOR BIG CYPRESS


sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

batch_size = 1
buffer = math.floor(len(df)/seq_length)
buffer


dataset = (
    dataset
    .shuffle(buffer)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)




