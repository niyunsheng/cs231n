# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt



# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds=np.split(X_train,num_folds)
y_train_folds=np.split(y_train,num_folds)

Xval_rows = X_train[:1000, :] # take first 1000 for validation
Yval = y_train[:1000]
Xtr_rows = X_train[1000:, :] # keep last 49,000 for train
Ytr = y_train[1000:]

k_to_accuracies = {}

from cs231n.classifiers import KNearestNeighbor

for k in k_choices:
    nn = KNearestNeighbor()
    nn.train(Xtr_rows,Ytr)
    Yval_predict = nn.predict(Xval_rows, k = k)
    k_to_accuracies[k]=np.mean(Yval_predict == Yval)
    print('k = %d, accuracy = %f' % (k, k_to_accuracies[k]))
    