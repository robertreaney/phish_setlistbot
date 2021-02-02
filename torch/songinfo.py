import pickle
import pandas as pd
import numpy as np

class SongInfo():
    """
    This class handles tracking of gap statistic and a eventually a boolean for if its a cover song.
    Needs a dataframe with 'Song' and 'Date' to calculate gap.
    """
    def __init__(self, data=None, PATH_TO_ORIGINALS=None): #allow option to not instantiate with both of these
        """
        Takes the dataset being used by the model.
        Args: data as a pandas dataframe or path to a csv with a 'Song' column. Also takes path to originals.txt for IsCover comparison
        """
        #reference point for calculating current gap
        if data != None:
            if type(data) != pd.core.frame.DataFrame:
                try:
                    data = pd.read_csv(data) #if data entered is a path
                except:
                    "Gap Calculator requires path to data or a pandas dataframe"
            else:
                current_date = data.show.to_numpy()[-1]
                unique_songs = set(data.Song)
                self.gap = dict()
                for song in unique_songs:
                    most_recent = data[data.Song == song].show.to_numpy()[-1]      
                    self.gap[song] = current_date - most_recent #initialize dictionary with current gap
       
        #get other songinfo -> originals
        if PATH_TO_ORIGINALS != None:
            with open (PATH_TO_ORIGINALS, 'rb') as fp:
                self.originals = pickle.load(fp)

    def find_gaps(self, data): #allow user to enter a path or pandas df
        if type(data) != pd.core.frame.DataFrame:
            try:
                data = pd.read_csv(data) #if data entered is a path
            except:
                "Gap Calculator requires path to data or a pandas dataframe"
        else:
            current_date = data.show.to_numpy()[-1]
            unique_songs = set(data.Song)
            self.gap = dict()
            for song in unique_songs:
                most_recent = data[data.Song == song].show.to_numpy()[-1]      
                self.gap[song] = current_date - most_recent #initialize dictionary with current gap

    def get_gap(self, song):
        assert song in self.song_info.keys(), "No data contained for song: %s"%song
        return gap[song]

    def update_gap(self, newshow): #maybe we want to put in a show and update this dict
        for ii, item in enumerate(self.gap):
            self.gap[item] += 1
        for song in newshow:
            self.gap[song] = 0

######## need area for the IsCover check
    def load_originals(self, PATH_TO_ORIGINALS): #allow user to input this on instantiation but also to update this with method
        with open (PATH_TO_ORIGINALS, 'rb') as fp:
            self.originals = pickle.load(fp)

    def update_originals(self, newsong):
        for song in list(newsong):  #allow user to input single string or a list or string
            self.originals.append(song)

    def isCover(self, song):
        if song in self.originals:
            return False
        else:
            return True

 if __name__ == '__main__':
    main()