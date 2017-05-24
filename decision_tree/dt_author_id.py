#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

classifierA = tree.DecisionTreeClassifier(min_samples_split=40)
classifierA.fit(features_train, labels_train)

# # Accuracy is 0.9789
print "Accuracy:", classifierA.score(features_test, labels_test)

# There are 3785 features
# print len(features_train[0])

# changing SelectPercentile to 1 instead of 10
# drops the number of features down to 379
# It looks like the SelectPercentile is used to select the percentage
# of features to keep based on their score (probably based on how useful they seem)
# A large percentile would make for a much more complex decision tree
# because of the increased number of features to split on

# changing the Percentile to 1 makes the fitting much faster
# The accuracy becomes 0.9664

