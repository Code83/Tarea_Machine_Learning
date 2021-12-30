#%%
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

#import seaborn as sb

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

X = datos[['Sesgo','V2']].values
y = datos['V5'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

n_neighbors= 10

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Precisión de K-NN classifier en entrenamiento set: {:.2f}'.format(knn.score(X_test , y_test)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# %%
