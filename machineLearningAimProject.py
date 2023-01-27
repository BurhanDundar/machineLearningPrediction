#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import cluster,datasets

# ## 4.1.d - Wine Dataset Decision Tree Classification

wines = load_wine() # load wine dataset

# seperate wine data and target 
X = wines.data 
y = wines.target

wines.target_names

wines.feature_names

clf = DecisionTreeClassifier(random_state=0)

# train the model seperating data as train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size =
0.3, random_state = 0, stratify = y)

# prediction with trained model
clf.fit(X_train,y_train)
test_sonuc = clf.predict(X_test)
print(test_sonuc)

# ## 4.1.e

# create confusion matrix 
cm = confusion_matrix(y_test,test_sonuc)
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# print cross validation score
crossValScore = cross_val_score(clf,wines.data,wines.target,cv=10)
print(crossValScore)

predicted = cross_val_predict(clf, wines.data, wines.target, cv=10)
print("accuracy",accuracy_score(wines.target, predicted))
print("precision score",precision_score(wines.target, predicted, average='macro') ) 
print("recall score",recall_score(wines.target, predicted, average='macro') ) 

# ## 4.1.f

test = ([2,0,2,1,1,2,1,2,0,2,2,2,2],
        [1.31, 2, 2.140e+00, 1.120e+01, 1.000e+02, 2.650e+00,
        2.760e+00, 2.6, 1.280e+00, 3, 1, 3.400e+00,
        1.050e+03])
test_sonuc = clf.predict(test)
print(test_sonuc)

# ## 4.2.d - Wine Dataset K-Means (Clustering) Classification

wines = load_wine()
X_wine = wines.data
y_wine = wines.target

# clustering on data models using k-means,with 3 clusters
k_means = cluster.KMeans(n_clusters=3)
k_means.fit_predict(X_wine)
y_pred = k_means.predict(X_wine)

#kmeans labels
print(k_means.labels_[::10])

# data target labels
print(y_wine[::10])

# ## 4.2.e

cm2 = confusion_matrix(y_wine,k_means.labels_)
print("ConfusionMatrix",cm2)

# Show confusion matrix in a separate window
plt.matshow(cm2)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# print accuracy,precision and recall score
predicted = cross_val_predict(k_means,wines.data , wines.target, cv=10)
print("accuracy score: ",accuracy_score(wines.target, predicted))
print("precision score: ",precision_score(wines.target, predicted, average='macro')) 
print("recall score: ",recall_score(wines.target, predicted, average='macro')) 

# ## 4.2.f

X =  [[0,1,5,3,4,1,2,3,4,3,1,2,2],[2,2,2,2,2,2,2,2,2,2,2,2,2],[2,1,3,3,4,2,2,2,2,1,2,8,2]]
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X)
print(k_means.labels_)




