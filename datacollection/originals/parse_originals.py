file = open('originals/originals.txt', mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()
#separate file by newline and save as a string in a list
my_list = []
for line in lines:
    line = line.split('\n')
    line = [i.strip() for i in line]
    my_list.append(line)

songs = []
for ii in my_list:
    songs.append(ii[0].split("\t")[0])

import pickle
with open('originals_cleaned.txt', 'wb') as fp:
    pickle.dump(songs, fp)

# ####to load with pick do this
# import pickle
# with open ('originals_cleaned.txt', 'rb') as fp:
#     list_1 = pickle.load(fp)