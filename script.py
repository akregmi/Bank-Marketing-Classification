#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:41:11 2022

@author: aayush
"""

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Create function to streamline preprocessing
def preprocessing(data):
    # Print size before preprocessing
    print("Size before preprocessing:",data.shape)
    
    # Remove entries with unknown values
    data = data[data.job != 'unknown']
    data = data[data.education != 'unknown']
    #data = data[data.contact != 'unknown']
    #data = data[data.poutcome != 'unknown']
    
    # Reshape the data
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    data['default'] = data['default'].map({'yes': 1, 'no': 0})
    data['housing'] = data['housing'].map({'yes': 1, 'no': 0})
    data['loan'] = data['loan'].map({'yes': 1, 'no': 0})
    data = pd.get_dummies(data, columns=['job','marital','education','contact','month','poutcome'])
    
    
    # Print size after precossing and return reshaped data
    print("Size after preprocessing:",data.shape)
    print("-"*40)
    return(data)
    

# Read and set up data into X and Y values
data = pd.read_csv('./bank-full.csv', sep=';')
data = preprocessing(data)
X = data.drop(['y'], axis=1)
y = data['y']

# Fit Kmeans clustering on training set
y.value_counts()
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
kmeans.labels_
print("Result of KNN on dataset: \n",classification_report(y, kmeans.labels_, target_names=['no','yes']))
print("SSE of samples to closest centroid: ", kmeans.inertia_)

'''
# Attempt to use DBScan to cluster instances

X = StandardScaler().fit_transform(X)
db = DBSCAN(eps=1)
db.fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
'''
