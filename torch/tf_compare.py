|


######################
embedding_dim = 5
hidden_nodes = 256
hidden_layers = 2
sequence_length = 300
batch_size = 64
learning_rate = .08



# compare outputs to tf

#data import
data = pd.read_csv("C:/Users/rober/Documents/projects/phishbot/datacollection/final_nosoundchecks.csv")

#data = pd.read_csv("C:/Users/rober/Documents/projects/phishbot/datacollection/final.csv")
#df = data[(data.date > "2009-01-01") & (data.date < "2019-01-01")].copy()
df = data[(data.date > "2009-01-01") & (data.date < "2019-01-01")].Song.copy()
#df = data[data.date < "2019-01-01"].copy()
#test = data[data.date > "2019-02-20"]
test = data[(data.date > "2019-05-01") & (data.date < "2019-08-15")]

# test = []
# test.append(data[data.date == "2019-07-12"].Song)
# test.append(data[data.date == "2019-07-13"].Song)
# test.append(data[data.date == "2019-07-14"].Song)
# test.append(data[data.date == "2019-08-30"].Song)
# test.append(data[data.date == "2019-08-31"].Song)
# test.append(data[data.date == "2019-09-01"].Song)
# val = data[data.date > "2019-07-11"].Song

#data3point0 = data[(data.date > "2009-01-01") & (data.date < "2019-07-12")]

# train = data[(data.date > "2009-01-01") & (data.date < "2019-01-01")].Song
# val = data[data.date >= "2019-01-01"].Song


#df = data3point0['Song'].copy()
del data#, data3point0

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
#val_ids = translate.ids_from_songs(list(val))

#val_ids
#all_ids
# np.savetxt("raydata/val.csv", val_ids.numpy(), delimiter=",")
# np.savetxt("raydata/train.csv", train_ids.numpy(), delimiter=",")

#x, y = input_label_split(train_ids, seq_length=sequence_length, overlap_inputs=False)
x, y = data_split(train_ids, sequence_length=sequence_length)
#xv, yv = data_split(val_ids, sequence_length=sequence_length)

ds = torch.utils.data.TensorDataset(x,y)
dataset = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# dsv = torch.utils.data.TensorDataset(xv, yv)
# datasetv = torch.utils.data.DataLoader(dsv, batch_size=batch_size, shuffle=True)


#xb, yb = next(iter(dataset))  #(batchsize x sequence_length), batchsize
#xb.shape


model = networks.NextNet(vocab_size, hidden_nodes, hidden_layers)
model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_f = nn.CrossEntropyLoss()

#predsb = model(xb) #LOGITS

#######
evaluation.fit(model, opt, loss_f, dataset, epochs=40)


#model.fit(opt, loss_f, dataset, datasetv, epochs=10)  #moving fit outside of the model class because itll be cleaner

###### make predictions
#1) make predictions for next show
#2) update inputs with new show
#3) repeat
eval_feed = df[-sequence_length:] 
evals = evaluation.NextNetEvaluator(model, translate)

winners = []
for ii in range(5):
    newshow = evals.predict(eval_feed)
    succeses, number = evals.eval_preds(newshow, test[test.date == "2019-06-11"])
    winners.append(number)
statistics.mean(winners) #average number of winners across 25 sets of predictions for the show

evals.update_with_show(data[data.date == "2019-09-01"].Song)


#### 

test = test[test.date != '2019-02-20']

for ii in sorted(list(set(test.date))):
    data[data.date == ii] #grab data for each future show iterativeley
    


####################################### save model
state = {
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
}
savepath='tuning/fittedmodel.t7'
torch.save(state,savepath)
########################load model
embedding_dim = 5
hidden_nodes = 256
hidden_layers = 2
sequence_length = 300
batch_size = 64
learning_rate = .08

##
data = pd.read_csv("C:/Users/rober/Documents/projects/phishbot/datacollection/final_nosoundchecks.csv")
df = data[(data.date > "2009-01-01") & (data.date < "2019-01-01")].Song.copy()
test = data[(data.date > "2019-05-01") & (data.date < "2019-08-15")]

translate = Vocab(df)
vocab_size = len(translate.vocab_dict) 
#
model = networks.NextNet(vocab_size, hidden_nodes, hidden_layers)
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

checkpoint = torch.load('C:/Users/rober/Documents/projects/phishbot/torch/tuning/fittedmodel.t7')
model.load_state_dict(checkpoint['state_dict'])
opt.load_state_dict(checkpoint['optimizer'])
#
model.eval()
eval_feed = df[-sequence_length:] 
evals = evaluation.NextNetEvaluator(model, translate)

winners = []
for ii in range(5):
    newshow = evals.predict(eval_feed)
    succeses, number = evals.eval_preds(newshow, test[test.date == "2019-06-11"])
    winners.append(number)
statistics.mean(winners) #average number of winners across 25 sets of predictions for the show

##
