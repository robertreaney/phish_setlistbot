import torch
import pandas as pd
class Vocab():
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
        self.vocab_dict = {song: (i+2) for i, song in enumerate(songs)}
        self.vocab_dict[''] = 0
        self.vocab_dict['Unknown'] = 1
        #make id dict
        self.id_dict = dict()
        for song in self.vocab_dict:
            self.id_dict[self.vocab_dict[song]] = song

    def ids_from_songs(self, data):
        """
        Function to retreive ids given songs.
            Args: song or list of songs
        """
        newlist = True
        for ii in data: #this is 100x faster as a tensor so dont clean this up and make it a list
            if newlist: #tensor wont allow you to append an empty list so this janky shit will suffice for creation
                try:
                    id_series = torch.tensor([self.vocab_dict[ii]])
                except:
                    id_series = torch.tensor([1])
                newlist=False
            else:
                try:
                    #id_series.append(self.vocab_dict[ii])
                    id_series = torch.cat((id_series, torch.tensor([self.vocab_dict[ii]])))
                except:
                    id_series = torch.cat((id_series, torch.tensor([1])))
                    #id_series.append(1) #in case item is not in our dictionary, return 0       
        return id_series

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
            try:
                song_series.append(self.id_dict[ii])
            except:
                song_series.append('Unknown')
            # try:
            #     song_series.append(self.id_dict[ii])
            # except:
            #     song_series.append("Unknown")
        if want_list:
            return song_series
        else:
            return song_series[0]
            

if __name__ == '__main__':
    main()