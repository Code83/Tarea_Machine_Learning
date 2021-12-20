import pandas as pad
import numpy as np
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split

df = pad.read_csv('data/data_banknote_authentication.txt', sep=",", header=None)
df.head()

print(df)
#print(df.shape)
#print(df.columns)
#print(df.info())