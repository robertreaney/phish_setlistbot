import pandas as pd
import numpy as np
from pathlib import Path
import os

p = Path('./Documents/GitHub/phish_setlistbot/datacollection/setlists/3')


filepaths = list(p.glob('**/*.csv'))

df = []

for item in filepaths:
    df.append(pd.read_csv(item))

df = pd.concat(df)
df
df.write_csv("test.csv")

pd.read_csv()