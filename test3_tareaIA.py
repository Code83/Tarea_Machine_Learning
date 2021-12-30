#%%
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
#import seaborn as sb

#%matploitlib inline

plt.rcParams['figure.figsize'] = (16 ,9)
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

datos = pd.read_csv('data/data_banknote_authentication.txt', sep=',')
datos.hist()
plt.show()

X = datos['V1'].values
y = datos['V5'].values
# %%
