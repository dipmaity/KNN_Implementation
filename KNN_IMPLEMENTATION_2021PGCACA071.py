from sklearn import *
#using the fit function from sklearn 
from sklearn.datasets import load_wine
wine_dataset = load_wine()
X = wine_dataset.data
Y = wine_dataset.target
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
knn.predict(X_test)
knn.score(X_test, Y_test)



# implementation without sklearn
import numpy as np
# create a function named as distance which return the Euclidean distance
def distance(t, n) :
  sum = np.sum(np.square(t - n))
  return np.sqrt(sum)
# Predict a value from X_test 
size = len(X_test)
prec = []
for i in range(size) :
  predict_x = X_test[i]
  actual_y = Y_test[i]
  v = []
  # find all distance from any instance of X_train to our predict value and add all distance in an array v
  for i, t in enumerate(X_train) :
    d = distance(t, predict_x)
    v.append(d)


  v = np.array(v)
  # Sort the array in increasing fashion
  ind = np.argsort(v)
  # let assume k for knn neighbour
  k = 3

  siz = len(np.unique(Y))
  arr = np.zeros(siz)
  # now check for which class have more closer to the predicted point
  for x in range(k):
    arr[Y_train[ind[x]]] += 1
  # print the predicted class
  ans = np.argsort(arr)
  prec.append(ans[k - 1])

prec = np.array(prec)
print(prec)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, prec)
print(cm)