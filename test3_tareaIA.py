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

X = datos[['Varianza','Sesgo']].values
y = datos['Clase'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

n_neighbors= 5

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Precisión de K-NN classifier en entrenamiento set: {:.2f}'.format(knn.score(X_test , y_test)))
print('Precisión of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

#Acá empezamos a graficar por vecino (K)
training_accuracy = []
test_accuracy = []
h = .02 # tamaño de la muestra

#Instancia de vecinos para clasificar  y el tamaño de la muestra

clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

training_accuracy.append(clf.score(X_train, y_train))
test_accuracy.append(clf.score(X_test, y_test))

#Punto de unión 

#x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
#y_min, y_max = X[:,0].min() - 1, X[:,0].max() + 1

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('K')
plt.ylabel('Precision')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])    
# %%
