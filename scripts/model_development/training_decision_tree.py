# -*- coding: utf-8 -*-
"""
@author: sushil
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import joblib


# Change this folder path with the path where the repo is cloned
root_folder = 'E:/tide/receipt_match'

## Read Data
df = pd.read_csv(root_folder + '/data/data_interview_test.csv', sep=':')
# Create Target 
df['target'] = np.where(df['matched_transaction_id']==df['feature_transaction_id'], 1 , 0)

## Train test split
feature_names = ['DateMappingMatch', 'TimeMappingMatch', 'PredictedTimeCloseMatch',
        'DescriptionMatch', 'ShortNameMatch', 'PredictedNameMatch',
        'AmountMappingMatch', 'PredictedAmountMatch']
X = df[['DateMappingMatch', 'TimeMappingMatch', 'PredictedTimeCloseMatch',
        'DescriptionMatch', 'ShortNameMatch', 'PredictedNameMatch',
        'AmountMappingMatch', 'PredictedAmountMatch']].values
Y = df[['target']].values.reshape(-1,)
ss = StratifiedShuffleSplit(n_splits=1,random_state=2022,test_size=0.2)
for train_index, test_index in ss.split(X,Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

## Oversample to tackle class imbalance
oversample = RandomOverSampler(sampling_strategy='minority')
X_train_balanced, Y_train_balanced = oversample.fit_resample(X_train, Y_train)
    

# Initialization of DT estimator
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, min_impurity_decrease=0.0007)
# Fitting the estimator on training data
dt.fit(X_train_balanced, Y_train_balanced)

# Model Performance on training data
y_train_predicted = dt.predict(X_train_balanced)
train_confusion_matrix = confusion_matrix(Y_train_balanced, y_train_predicted)
print('Training Confusion Matrix:')
print(pd.DataFrame(train_confusion_matrix, index=['0', '1'], columns=['0', '1']))
print('Training AUC: ' + str(round(roc_auc_score(Y_train_balanced, dt.predict_proba(X_train_balanced)[:, 1]), 4)))
print('Training Recall: ' + str(round(recall_score(Y_train_balanced, y_train_predicted), 4)))
print('\n\n')

# Model Performance on testing data
y_test_predicted = dt.predict(X_test)
test_confusion_matrix = confusion_matrix(Y_test, y_test_predicted)
print('Testing Confusion Matrix:')
print(pd.DataFrame(test_confusion_matrix, index=['0', '1'], columns=['0', '1']))
print('Testing AUC: ' + str(round(roc_auc_score(Y_test, dt.predict_proba(X_test)[:, 1]), 4))) 
print('Testing Recall: ' + str(round(recall_score(Y_test, y_test_predicted), 4)))


## Create Decision tree visualization using Ghaphviz
graph = export_graphviz(
        dt,
        out_file=None,
        feature_names=feature_names,
        class_names=['NotMatch','Match'],
        rounded=True,
        filled=True)   

g = graphviz.Source(graph)
g.view()

## Saving the model
# Uncomment below line to save a new model
# joblib.dump(dt, root_folder + '/trained_models/dt_receipt_match_model.pkl')
