class SongInfo():
    """
    This class handles tracking of gap statistic and a eventually a boolean for if its a cover song.
    Needs a dataframe with 'Song' and 'Date' to calculate gap.
    """
    def __init__(self, data):
        """
        Takes the dataset being used by the model
        """
        #reference point for calculating current gap
        current_date = df.show.to_numpy()[-1]
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

