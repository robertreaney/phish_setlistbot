import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
import math
import os
import time

embedding_dim = 1000
rnn_units = 1000
seq_length = 200

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


data = pd.read_csv("datacollection/final.csv")
#data.head()

test = []
test.append(data[data.date == "2019-07-12"].Song)
test.append(data[data.date == "2019-07-13"].Song)
test.append(data[data.date == "2019-07-14"].Song)
test.append(data[data.date == "2019-08-30"].Song)
test.append(data[data.date == "2019-08-31"].Song)
test.append(data[data.date == "2019-09-01"].Song)

data3point0 = data[(data.date > "2009-01-01") & (data.date < "2019-07-12")]

df = data3point0['Song'].copy()

del data, data3point0

#df.to_csv("3point0train.csv", index=False)


##### just look at song sequences and pop out next song
#find number of unique songs
vocab = sorted(set(df))
len(vocab) #953 unique songs

#convert strings to numeric
ids_from_songs = preprocessing.StringLookup(vocabulary=list(vocab))

ids = ids_from_songs(["Possum", "Cities"]) #returns numeric representation of a song
ids #outputs tf.Tensor object with ids

songs_from_ids = preprocessing.StringLookup(vocabulary=ids_from_songs.get_vocabulary(), invert=True)
songs = songs_from_ids(ids) #takes in tf.Tensor object of ids to output songs
len(songs)
#tf function to convert back to string from nyte
tf.compat.as_str_any(songs.numpy()[0])

def songlist_from_ids(ids): #this function takes tf object of ids and outputs setlist
    songlist = []
    songs = songs_from_ids(ids)
    try:
        for ii in range(len(ids)):
            songlist.append(tf.compat.as_str_any(songs.numpy()[ii]))
    except: #dont make a list if there is only one song. messy output
        return tf.compat.as_str_any(songs.numpy())
    return songlist

songlist_from_ids(ids) #test it and it works!

######
# PREDICTION TASK
######

#we need to break large song dataset into huge overlapping chunks to create "memory"
all_ids = ids_from_songs(df) #converts every song played to numeric vocab

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)



for ids in ids_dataset.take(10):
    print(songlist_from_ids(ids))


examples_per_epoch = len(df)//(seq_length+1)
examples_per_epoch
#convert songdf to tf sequences
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(2):
    print(seq)
    #print(songlist_from_ids(seq))
    
type(sequences)

####
#training
#####

#we need dataset of (input, label) pairs where each of these are sequences
#each time step input is the current song and label is the next song
#now take each sequence and make 1:n-1 the input and 2:n the output
def split_input_target(sequence):
    input_songs = sequence[:-1]
    target_songs = sequence[1:]
    return input_songs, target_songs

split_input_target(["yem", "possum", "waiting all night"])

#convert dataset to split sequences like (input, output)
dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print("Input: ", songlist_from_ids(input_example))
    print("Target: ", songlist_from_ids(target_example))
len(df)


#len(df)/100
#create training batches
#shuffle data and pack into batches
batch_size = 1
buffer = math.floor(len(df)/seq_length)
buffer


dataset = (
    dataset
    .shuffle(buffer)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



for ii in dataset.take(1):
    print(songs_from_ids(ii[0]))
####
#build the model
####
vocab_size = len(vocab)
vocab_size #number of unique songs


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state = states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


model = MyModel( #initialize network
    vocab_size=len(ids_from_songs.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
    )


### try the model
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)") 



model.summary()

#check out getting a prediction
sampled_indicies = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indicies = tf.squeeze(sampled_indicies,axis=-1).numpy()
sampled_indicies

#decode these to see the next sequence predicted by the model
songlist_from_ids(input_example_batch[0])
songlist_from_ids(sampled_indicies)

#############################
#### TRAIN THE MODEL!!
#######################
#attach optimizer and loss function
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)

#check if initialization is trash or not
tf.exp(mean_loss).numpy()
len(vocab)

#configure training procedure
model.compile(optimizer='adam', loss=loss)

#configure checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only=True
)

#execute training
EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


######
# utilize model
######

#make a single step prediction
class OneStep(tf.keras.Model):
  def __init__(self, model, ids_from_songs, songs_from_ids, temperature=1.0):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.songs_from_ids = songs_from_ids
    self.ids_from_songs = ids_from_songs

    # # Create a mask to prevent "" or "[UNK]" from being generated.
    # skip_ids = self.ids_from_songs(['','[UNK]'])[:, None]
    # sparse_mask = tf.SparseTensor(
    #     # Put a -inf at each bad index.
    #     values=[-float('inf')]*len(skip_ids),
    #     indices = skip_ids,
    #     # Match the shape to the vocabulary
    #     dense_shape=[len(ids_from_songs.get_vocabulary())]) 
    # self.prediction_mask = tf.sparse.to_dense(sparse_mask)


  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_ids = self.ids_from_songs(inputs)#.to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits] 
    predicted_logits, states =  self.model(inputs=input_ids, states=states, return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
    #predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_song = self.songs_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_song, states



one_step_model = OneStep(model, songs_from_ids, ids_from_songs)





###
def doPred(next_song, states):
    input_ids = ids_from_songs(next_song)#.to_tensor()
    input_ids

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits] 
    predicted_logits, states =  model(inputs=input_ids, states=states, 
                                            return_state=True)
        # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    #predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        #predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
    predicted_logits
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
    predicted_song = songs_from_ids(predicted_ids)
    return predicted_song, states


states = None
pred_base = np.asarray(df[-100:]) #grab recent songs
next_show = []
for n in range(30):
    song, states = doPred(tf.constant([pred_base]), states) #output next song
    next_show.append(song.numpy()[0].decode("utf-8"))
    pred_base = np.append(pred_base, np.asarray(song.numpy())) #add predicted song to recent songs for next pred


for song in next_show:
    print(song)


test.Song
states = None

####see how it did
def eval_preds(next_show, test):
    correct = 0
    correct_preds = []
    for song in set(next_show):
        for target in set(test):
            if song == target:
                correct += 1
                correct_preds.append(song)
    return correct, correct_preds
correct, correct_preds = eval_preds(next_show, test[0])
correct
correct_preds

test.Song
next_show[0] in test.Song

##### loop through multiple shows to test
states = None
results = dict()
for ii, item in enumerate(test):
    states = None
    if ii < 5:
        pred_base = np.asarray(df[-(100-(ii*20)):]) #grab recent songs
        if ii > 0:
            for jj in range(ii):
                pred_base = np.append(pred_base, test[jj])
    elif ii == 5:
        pred_base = []
        for jj in range(5):
            pred_base = np.append(pred_base, test[jj])
    else:
        pred_base = []
        for jj in [1,2,3,4,5]:
            pred_base = np.append(pred_base, test[jj])

    next_show = []
    for n in range(30):
        song, states = doPred(tf.constant([pred_base]), states) #output next song
        next_show.append(song.numpy()[0].decode("utf-8"))
        pred_base = np.append(pred_base, np.asarray(song.numpy())) #add predicted song to recent songs for next pred

    correct, correct_preds = eval_preds(next_show, test[ii])
    results[ii] = [correct, correct_preds]

    results[6]