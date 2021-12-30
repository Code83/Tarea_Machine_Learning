#%%
import pandas as pad
import numpy as np
import mglearn
from IPython.display import display
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

<<<<<<< HEAD
dataSet = pad.read_csv('data/data_banknote_authentication.txt', sep=",", header=None)
x, y = mglearn.datasets.make_forge()
#X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, stratify=dataSet.target, random_state=66)

X_train, X_test, y_train, y_test = train_test_split(x, y)
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)
=======
data = pad.read_csv('data/data_banknote_authentication.txt', sep=",")
#X, y = mglearn.datasets.make_forge()
X = data['V1'].values
y = data['V2'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66)
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 20)
>>>>>>> 8d9e164a2a68b6920eb7ce3da1f664d863a918ac
for n_neighbors in neighbors_settings:
  # build the model
  clf = KNeighborsClassifier(n_neighbors=n_neighbors)
  clf.fit(X_train, y_train)
  # record training set accuracy
  training_accuracy.append(clf.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(clf.score(X_test, y_test))
<<<<<<< HEAD
plt.plot(neighbors_settings, training_accuracy, label="training precisión")
plt.plot(neighbors_settings, test_accuracy, label="test precisión")
plt.ylabel("precisión")
plt.xlabel("vecinos")
=======
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Precisión")
plt.xlabel("Vecinos")
>>>>>>> 8d9e164a2a68b6920eb7ce3da1f664d863a918ac
plt.legend()
print("Test set precisión: {:.2f}".format(clf.score(X_test, y_test)))
print("Test set predicción: {}".format(clf.predict(X_test)))

# %%

# %%

# %%
