#%%
import pandas as pad
import numpy as np
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

dataSet = pad.read_csv('data/data_banknote_authentication.txt', sep=",", header=None)
x, y = mglearn.datasets.make_forge()
#X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target, stratify=dataSet.target, random_state=66)

X_train, X_test, y_train, y_test = train_test_split(x, y)
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
  # build the model
  clf = KNeighborsClassifier(n_neighbors=n_neighbors)
  clf.fit(X_train, y_train)
  # record training set accuracy
  training_accuracy.append(clf.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training precisión")
plt.plot(neighbors_settings, test_accuracy, label="test precisión")
plt.ylabel("precisión")
plt.xlabel("vecinos")
plt.legend()
print("Test set precisión: {:.2f}".format(clf.score(X_test, y_test)))
print("Test set predicción: {}".format(clf.predict(X_test)))

# %%

# %%

# %%
