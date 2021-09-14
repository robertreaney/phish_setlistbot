import requests as r
import json
from html.parser import HTMLParser
from datetime import datetime
from tqdm import tqdm
import concurrent.futures as cf

PARAMETERS = {
    'net_key' : "B93C5925297AB520755D",
    'net_url' : f"https://api.phish.net/v3",
    'in_key' : '10ad07b2f96ab003b640192eb04590eb7ceb0ef4f6cb86a38308aede051d9960704857dc697326c1d49d99ed922ab6ad',
    'in_url' : 'http://phish.in/api/v1/'
}



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


class ApiWrapper():
    def __init__(self, PARAMETERS):
        self.parser = MyParser()
        self.data = PARAMETERS

    def get_shows(self, per_page, page):
        message = r.get(self.data['in_url'] + '/shows',
            headers={
                'Authorization': f'Bearer {self.data["in_key"]}',
                'Accept': 'application/json'
                },
            params={
                'sort_attr': 'date',
                'sort_dir': 'desc',
                'per_page': per_page,
                'page': page
        })
        return message

    def get_eras(self):
        response = r.get(f"{self.data['in_url']}/eras", headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {self.data['in_key']}"
        })
        self.eras = response.json()['data']
        return self.eras
    
###############################################################
class DataManager(ApiWrapper):
    def get_showdata(self, start_date):
        self._get_showdates(start_date)
        self.data['setlists'] = {x: self.get_setlist(x) for x in self.data['show_dates']}

    def get_setlist(self, date):
        """Returns showdata if there is a show, returns None is there isn't
        args:
            date: str - showdate
        """
        response = r.get(f"{self.data['net_url']}/setlists/get?apikey={self.data['net_key']}", \
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

    # private
    def _datetime(self, date):
        """return datetime object from a date like -> '2018-11-03'
        """
        return datetime(*[int(x) for x in date.split('-')])

    def _get_showdates(self, start_date):
        """populate self.data['show_dates'] with all shows going back to supplied start_date arg."""
        # loop through the others until we reach the end date
        self.data['show_dates'] = []
        message = self.get_shows(100, page=1)
        self.data['show_dates'] = [x['date'] for x in message.json()['data'] if 
            self._datetime(x['date']) > self._datetime(start_date)]
        total_pages = message.json()['total_pages']
        current_page = 2
        with tqdm(total=total_pages, desc='Getting showdates.') as pbar:
            pbar.update(1)
            while current_page <= total_pages:
                message = self.get_shows(100, page=current_page)
                self.data['show_dates'] += [x['date'] for x in message.json()['data'] if 
                    self._datetime(x['date']) > self._datetime(start_date)]
                current_page += 1
                if self._datetime(message.json()['data'][-1]['date']) < self._datetime(start_date):
                    print("\nHit start_date, terminating early!")
                    break
                pbar.update(1)

dm = DataManager(PARAMETERS)
# dm.get_showdates('2018-11-03')
# dm.data['show_dates']
# dm.get_setlist('2018-11-03')
dm.get_showdata('2009-01-01')
