import requests as r
import json
from html.parser import HTMLParser
import datetime
from tqdm import tqdm


key = "B93C5925297AB520755D"
url = f"https://api.phish.net/v3"

##################################################################
# phish.in
in_key = '10ad07b2f96ab003b640192eb04590eb7ceb0ef4f6cb86a38308aede051d9960704857dc697326c1d49d99ed922ab6ad'
in_url = 'http://phish.in/api/v1/'




class MyParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refresh()
    def refresh(self):
        self.data = {
            "set": [],
            "song": []
        }
    def handle_starttag(self, tag, attrs):
        self._current_tag = tag
    def handle_data(self, data):
        if self._current_tag == 'span':
            if data != ": ":
                self._current_set = data
        elif self._current_tag == 'a':
            if all(x.isalpha() or x.isspace() or x.isnumeric() or x == "'" for x in data):
                self.data['song'].append(data)
                self.data['set'].append(self._current_set)
            # else:
            #     print(f"current segue: {data}")
            #     self.data['segue'].append(data.strip())



###############################################################
class DataManager():
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.parser = MyParser()
        self.shows = []

    def get_eras(self):
        response = r.get(f"{in_url}/eras", headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {in_key}"
        })
        self.eras = response.json()['data']
        return self.eras

    def get_showdates(self, past_years):
        years = .5
        today = datetime.date.today()
        start = today
        how_long = datetime.timedelta(days=365.25*years)
        delta = datetime.timedelta(days=1)

        # TODO 5it/s there has got to be an easier way to find out all the days with shows
        with tqdm(total=(today - (today-how_long)).days) as pbar:
            while today > start - how_long:
                today = today - delta
                date = today.strftime("%Y-%m-%d")
                if self.get_showdata(date) is not None:
                    self.shows.append(date)
                pbar.update(1)

    def get_showdata(self, date):
        """Returns showdata if there is a show, returns None is there isn't
        args:
            date: str - showdate
        """
        response = r.get(f"{self.url}/setlists/get?apikey={self.key}", \
            params={
            "showdate": date
            }).json()
        if response['error_code'] != 0:
            raise ValueError(f"Get setlist failure: {response['error_message']}")
        try:
            setlist = response['response']['data'].pop()
            # dict_keys(['showid', 'showdate', 'short_date', 
            # 'long_date', 'relative_date', 'url', 'gapchart', 
            # 'artist', 'artistid', 'venueid', 'venue', 'location', 
            # 'setlistdata', 'setlistnotes', 'rating'])
            
            weekday = setlist['long_date'].split()[0]
            venueid = setlist['venueid']
            location = setlist['location']

            setlistdata = setlist['setlistdata']
            self.parser.feed(setlistdata)
            setlistdata = self.parser.data
            self.parser.refresh()
            return {
                "weekday": weekday,
                "venueid": venueid,
                "location": location,
                "setlistdata": setlistdata
            }
        except:
            return None


dm = DataManager(url, key)
dm.get_showdates(3)
dm.shows
dm.get_showdata('2018-11-03') == None


