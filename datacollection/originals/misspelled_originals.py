#get originals
import pickle
with open ('originals/originals_cleaned.txt', 'rb') as fp:
    originals = pickle.load(fp)

#get all played songs in the proper strings for the rest of our analysis
import pandas as pd
data = pd.read_csv("C:/Users/rober/Documents/projects/phishbot/datacollection/final.csv")
songs = list(set(data.Song))

#we need a list of originals for the IsCover decision. So we need this [originals] list to have the correct spelling.
#Unforetunately there are a lot of abbreviated songs in the 'originals' list so we need to find whichs ones are abbreviated to write them correctly

solution = [] #somewhere to put the final originals
suspect = [] #somewhere to put the problem children
for song in originals:
    if song in songs:
        solution.append(song)
    else:
        suspect.append(song)

len(solution)
len(suspect) #only 33 problems!!!

suspect
fixed = ['Run Like an Antelope', 'A Song I Heard the Ocean Sing', 'All Things Reconsidered',
         'Big Black Furry Creature from Mars', 'Back on the Train', 'Carolina', 'Dog Log',
         'Gaul Swerves and the Rest is Everything Else', 'Glide II', 'Waking Up', 'Prince Caspian',
         'Back on the Train', 'Paul and Silas', 'Punch Me in the Eye', 'NICU', 'Keyboard Army',
         'Carini', 'Foam', 'Split Open and Melt', "My Mind's Got a Mind of its Own", 'NO2',
         "Olivia's Pool", 'Punch You in the Eye', 
         'Divided Sky', 'The Man Who Stepped Into Yesterday', 'Prep School Hippie', 'Time Turns Elastic',
         'In a Hole', 'Faht', 'You Enjoy Myself']

final = solution + fixed

#check to make sure these all are spelled correctly
problem = []
for ii in final:
    if ii in songs:
        continue;
    else:
        problem.append(ii)

problem
##########

#final is now a list of all original phish songs
import pickle
with open('originals_final.txt', 'wb') as fp:
    pickle.dump(final, fp)

# ####to load with pick do this
# import pickle
# with open ('originals_cleaned.txt', 'rb') as fp:
#     list_1 = pickle.load(fp)