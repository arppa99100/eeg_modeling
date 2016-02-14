import os
import sys
import pickle
import numpy as np
from dataio import readdata, readlabels, writedata

from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

path = os.path.abspath(os.path.join(__file__,"../"))
#Read data
train_dataset = readdata(path + "/curated/train_dataset")
train_labels = readlabels(path + "/curated/train_labels")
#valid_dataset = readdata(path + "/curated/valid_dataset")
#valid_labels = readlabels(path + "/curated/valid_labels")
test_dataset = readdata(path + "/curated/test_dataset")
test_labels = readlabels(path + "/curated/test_labels")
print("Training:", train_dataset.shape, train_labels.shape)
#print("Validation:", valid_dataset.shape, valid_labels.shape)
print("Testing:", test_dataset.shape, test_labels.shape)

#Reshape
n_input = 8*360 # EEG data input (8 channels * 1130 sample points)
n_classes = 5 # EEG total classes ("nothing", "up", "down", "left", "right")
train_dataset.shape = (train_dataset.shape[0], n_input)
test_dataset.shape = (test_dataset.shape[0], n_input)

#KNN
print("Training K-Nearest Neighbors...")
knn = KNeighborsClassifier(
        algorithm="auto", 
        weights="uniform", 
        n_neighbors=15)
knn.fit(train_dataset, train_labels)
knn_pred = knn.predict(test_dataset)
knn_acc = accuracy_score(test_labels, knn_pred)
print("Knn Acc: ", knn_acc)

#Random Forests
print("Training Random Forests...")
forest = ExtraTreesClassifier(n_estimators = 1000)
forest.fit(train_dataset, train_labels)
forest_pred = forest.predict(test_dataset)
forest_acc = accuracy_score(test_labels, forest_pred)
print("For Acc: ", forest_acc)

#SVM
print("Training Support Vector Machine...")
svm_mod = svm.LinearSVC(
  C=1.0,
  penalty="l2",
  loss="squared_hinge",
  tol=0.0001)
svm_mod.fit(train_dataset, train_labels)
svm_pred = svm_mod.predict(test_dataset)
svm_acc = accuracy_score(test_labels, svm_pred)
print("Svm Acc: ", svm_acc)

#Ada
print("Training Adaboost...")
ada = AdaBoostClassifier(n_estimators = 50)
ada.fit(train_dataset, train_labels)
ada_pred = ada.predict(test_dataset)
ada_acc = accuracy_score(test_labels, ada_pred)
print("Ada Acc: ", ada_acc)

#Logistic
print("Training Logistic Regression...")
lgr = linear_model.LogisticRegression()
lgr.fit(train_dataset, train_labels)
lgr_pred = lgr.predict(test_dataset)
lgr_acc = accuracy_score(test_labels, lgr_pred)
print("Lgr Acc: ", lgr_acc)

#Voting
print("Training Ensemble Classifier...")
vot = VotingClassifier(estimators=[
        ("KNN", knn),
        ("FOR", forest),
        ("SVM", svm_mod),
        ("ADA", ada),
        ("LGR", lgr)
      ], voting = "hard", weights=[6,5,1,1,1])
vot.fit(train_dataset, train_labels)
vot_pred = vot.predict(test_dataset)
vot_acc = accuracy_score(test_labels, vot_pred)
print("Ens Acc: ", vot_acc)

# Save the Classifier
with open(path + "/predicting/eeg.model", "wb") as writestream:
  pickle.dump(vot, writestream)
