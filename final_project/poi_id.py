#!/usr/bin/python

from __future__ import division
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np 
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'
features_list = [
    'poi',
    'salary',
    'total_payments', 
    'bonus', 
    'total_stock_value', 
    'expenses', 
    'exercised_stock_options', 
    'restricted_stock', 
    'to_messages', 
    'from_poi_to_this_person', 
    'from_messages', 
    'from_this_person_to_poi', 
    'shared_receipt_with_poi'
]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Data exploration
### Total number of data points
print ('Total number of data points: %d' %len(data_dict))

### Allocation across classes(POI/non-POI)
num_poi = 0
for name in data_dict:
    person = data_dict[name]
    if person['poi']:
        num_poi += 1
print ('Number of person of interest %d' %num_poi)
print ('Number of non person of interest %d' %(len(data_dict)-num_poi))

### Number of features used
names = data_dict.keys()
all_features = data_dict['METTS MARK'].keys()
print('Number of features used %d' %len(all_features))

### Missing values in features
def num_missing_value(feature):
    num_missing_value = 0
    for name in data_dict:
        person = data_dict[name]
        if person[feature] == 'NaN':
            num_missing_value += 1
    print ('Number of missing value in feature %s: %d' %(feature, num_missing_value))

num_missing_value('salary')
num_missing_value('to_messages')
num_missing_value('deferral_payments')
num_missing_value('total_payments')
num_missing_value('exercised_stock_options')
num_missing_value('bonus')
num_missing_value('restricted_stock')
num_missing_value('shared_receipt_with_poi')
num_missing_value('restricted_stock_deferred')
num_missing_value('total_stock_value')
num_missing_value('expenses')
num_missing_value('loan_advances')
num_missing_value('from_messages')
num_missing_value('from_this_person_to_poi')
num_missing_value('poi')
num_missing_value('director_fees')
num_missing_value('deferred_income')
num_missing_value('long_term_incentive')
num_missing_value('email_address')
num_missing_value('from_poi_to_this_person')

### Task 2: Remove outliers

### Remove total line
data_dict.pop('TOTAL', 0)

### Function to plot outliers
def PlotOutlier(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter( x, y )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

### Visualise outliers
#print(PlotOutlier(data_dict, 'salary', 'total_payments'))

from pprint import pprint

outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))

pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:3])

data_dict.pop('SKILLING JEFFREY K',0)
data_dict.pop('LAY KENNETH L',0)
data_dict.pop('FREVERT MARK A',0)

### Visualise data after removing outliers
#print(PlotOutlier(data_dict, 'salary', 'total_payments'))

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Add new features to dataset
from decimal import Decimal
def compute_ratio(num_poi_messages, num_total_messages):
    if num_poi_messages != 'NaN' and num_total_messages != 'NaN' and num_total_messages != 0:
        fraction = float(num_poi_messages / num_total_messages)
    else:
        fraction = 0
    return fraction
for name in my_dataset:
    name = my_dataset[name]
    percent_received_email_from_poi = compute_ratio(name['from_poi_to_this_person'], name['to_messages'])
    name['percent_received_email_from_poi'] = percent_received_email_from_poi
    percent_send_email_to_poi = compute_ratio(name['from_this_person_to_poi'], name['from_messages'])
    name['percent_send_email_to_poi'] = percent_send_email_to_poi
    print ('From person of interest to this person ratio: %f' % percent_received_email_from_poi)
    print ('From this person to person of interest ratio: %f' % percent_send_email_to_poi)

all_features = features_list + ['percent_received_email_from_poi', 'percent_send_email_to_poi']

print ('All features: {0}'.format(all_features))
print ('Number of features used %d' %len(all_features))

### Extract features and labels from dataset for local testing
### read in data dictionary, convert to numpy array
my_dataset = featureFormat(my_dataset, all_features)
labels, features = targetFeatureSplit(my_dataset)

### Scale features using MinMAxScaler
from sklearn.preprocessing import MinMaxScaler
scaled_features = MinMaxScaler().fit_transform(features)

### Deploy univariate feature selection with SelectKBest
from sklearn.feature_selection import SelectKBest
X, y = scaled_features, labels
k = 10
select_k_best = SelectKBest(k=k)
select_k_best.fit(X,y)
scores = select_k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
k_best_features = dict(list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))[:k])
print ('{0} best features: {1}'.format(k, k_best_features.keys()))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

### Lasso regression
from sklearn.linear_model import Lasso
lasso_clf = Lasso()

### K means
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Accuracy of naive bayes classifier
nb_clf.fit(features_train, labels_train)
nb_pred = nb_clf.predict(features_test)
from sklearn.metrics import accuracy_score
naive_bayes_test_score = accuracy_score(labels_test, nb_pred)
print ('Naive Bayes Classifier Test score %f' %naive_bayes_test_score)

### Accuracy of lasso regression classifier
#lasso_clf.fit(features_train, labels_train)
#pred = lasso_clf.predict(features_test)
#lasso_test_score = accuracy_score(labels_test, pred)
#print ('Test score %f' %lasso_test_score)

### Accuracy of KMeans classifier
k_clf.fit(features_train, labels_train)
k_pred = k_clf.predict(features_test)
k_means_test_score = accuracy_score(labels_test, k_pred)
print ('KMeans Classifier Test score %f' %k_means_test_score)

### K-Fold CV
#from sklearn.model_selection import KFold
#X = features
#y = labels
#kf = KFold(n_splits=2, shuffle = True)
#for train, test in kf.split(X):
#    X_train = [train]
#    X_test = [test]
#    print X_train

### Tunning naive bayes classifier
#from sklearn import grid_search
#parameters = {'priors': ['array-like']}
#nb_clf_tunned = grid_search.GridSearchCV(nb_clf, parameters)
#nb_clf_tunned.fit(features, labels)
#print nb_clf_tunned.best_params_

### Tunning lasso classifier

### Evaluation: Precision and recall
from sklearn.metrics import *
print precision_score(labels_test, nb_pred)
print recall_score(labels_test, nb_pred)
print precision_score(labels_test, k_pred)
print recall_score(labels_test, k_pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(nb_clf, my_dataset, k_best_features)
dump_classifier_and_data(k_clf, my_dataset, k_best_features)
