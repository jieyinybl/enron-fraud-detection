# Enron Fraud Detection
 
The project uses the enron dataset, which includes the email data and financial data of employees. The project uses machine learning algorithm to identify Enron Employees who might have committed fraud. 

**TODO**
While reading through the text you should highlight some variable / feature / algortihms, especially when it is a variable you defined your code. You can do this either:

- **makes it bold**
- `CodeStyle`

## Question 1:
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?
 
The dataset includes Enron employees’ email data and financial data. The project applies machine learning algorithm to discover patterns in the data in order to detect fraud. The dataset contains in total 146 data points, namely 146 employees data. The features in the dataset can be grouped into three categories: Financial features, Email features and POI label. 

First of all, the project takes a closer look at the dataset: total data points, allocation across classes, features used, and missing values. In the following a summary of the data exploration:

| Data                         | Value |
|:---------------------------- | -----:|
| Total number of data points: | 146 |
| Number of person of interest: | 18 |
| Number of non person of interest: | 128 |
| Number of features used | 19 |
| Number of missing value in feature salary: | 51 |
| Number of missing value in feature to_messages: | 60 |
| Number of missing value in feature deferral_payments: | 107 |
| Number of missing value in feature total_payments: | 21 |
| Number of missing value in feature exercised_stock_options: | 44 |
| Number of missing value in feature bonus: | 64 |
| Number of missing value in feature restricted_stock: | 36 |
| Number of missing value in feature shared_receipt_with_poi: | 60 |
| Number of missing value in feature restricted_stock_deferred: | 128 |
| Number of missing value in feature total_stock_value: | 20 |
| Number of missing value in feature expenses: | 51 |
| Number of missing value in feature loan_advances: | 142 |
| Number of missing value in feature from_messages: | 60 |
| Number of missing value in feature from_this_person_to_poi: | 60 |
| Number of missing value in feature director_fees: | 129 |
| Number of missing value in feature deferred_income: | 97 |
| Number of missing value in feature long_term_incentive: | 80 |
| Number of missing value in feature from_poi_to_this_person: | 60 |


Secondly, a two dimensional plotting will be applied to see if there is any outlier. By the first plotting, we plot the salary and total_payments and notice that there is an outlier ‘TOTAL’, which should be removed. 

**TODO insert a screenshot of the first plot**
![here should be the first plot](plot1.jpg)

After the ‘TOTAL’ is removed, we plot the salary and total_payments again and can see from the plot that there are three outliers in terms of salary.  

![here should be the second plot](plot2.jpg)

The three persons are printed out and they turn out to be the three big bosses of Enron at that time: 

```python
### Sort the list of outliers and print the top 3 outliers in the list
print ('Outliers in terms of salary: ')
pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[1:4])

### Print out the three outliers: persons with highest salary
print ('Print out the three outliers: Employees with highest salary:')
print (data_dict['SKILLING JEFFREY K'])
print (data_dict['LAY KENNETH L'])
print (data_dict['FREVERT MARK A'])
```
>[('SKILLING JEFFREY K', 1111258),
> ('LAY KENNETH L', 1072321),
> ('FREVERT MARK A', 1060932)]

As two of the three persons (SKILLING JEFFREY K & LAY KENNETH L) above are person of interest out of 18 person of interest in total, we will not removed these three outliers here.

## Question 2:
> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

The project creates 2 new features: `percent_received_email_from_poi` & `percent_send_email_to_poi`. As compared with absolute number, ratio would tells better how strong is the email connection of this person to the person of interest than to the non person of interest. After the two features are created, the project deploys univariate feature selection to select the k best features:

1. Create function with **SelectKBest** to select the k best features
1. Use **GridSearchCV** to select the best parameters
1. Set the parameters of the selected machine learning algorithms with the best parameters and evaluate the performance of the algorithms
1. Repeat the above process for k ranged from 3 to 10. Based on the Recall, select the best K value.

With the above process, k = 4 turns out to be the best K value. The selected features and the scores are:

| Feature | Score |
|:------- | -----:|
| bonus | 21.060001707536571 | 
| exercised_stock_options | 25.097541528735491 |
| salary | 18.575703268041785 | 
| total_stock_value | 24.467654047526398 |

As the features in the dataset includes both email features and financial features, whose range are quite different: for example, the max value for the feature salary is more than 10 million whereas the max value for the feature from_this_person_to_poi is less than 1k. In order to avoid that some features are too dominant due to its range, features will be rescaled with **MinMaxScaler**.

## Question 3:
> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [relevant rubric item: “pick an algorithm”]
 
**Naive bayes**, **support vector classifier**, **K nearest neighbors**, **decision tree** and **logistic regression** are tested with the selected k_best_features (k_range from 3 to 10). As evaluation strategy the accuracy_score, precision_score, recall_score and f1_score will be calculated.
To decide that which algorithms works better, the project will mainly focus on the recall. Moreover, the precision and f1_score which are greater than 0.3 will also be considered as good performance. The accuracy_score here is not a strong index for the performance, due to that there is only small percent of person of interest in the dataset. 

To sum up the evaluation strategy:

1. Different K value are tried with **SelectKBest**
1. Algorithms with precision_score, recall_score and f1_score greater than 0.3 are selected.
1. The higher the recall is, the better the performance of an algorithms is.

In the following an overview of the algorithms selected according to the above evaluation strategy. It turns out the with the k=4, the **Naive Bayes Classifier** shows the highest scores:

| K value | Classifier | Mean Accuracy | Mean of precision | Mean of recall | Mean of f1 score |
| ------- | ---------- | -------------:| -----------------:| --------------:| -----------:|
| k = 3 | Naive Bayes Classifier (without Tuning)         | 0.829 | 0.332 | 0.32  | 0.316919191919  |
| k = 4 | Naive Bayes Classifier (without Tuning)         | 0.861 | 0.511 | 0.42  | 0.42847041847   |
| k = 5 | Naive Bayes Classifier (without Tuning)         | 0.857 | 0.384 | 0.42  | 0.39271950272   |
| k = 5 | Logistic Regression Classifier (with Tuning)    | 0.877 | 0.434 | 0.3   | 0.343759018759  |
| k = 6 | Naive Bayes Classifier (without Tuning)         | 0.843 | 0.425 | 0.4   | 0.392395382395  |
| k = 7 | Naive Bayes Classifier (without Tuning)         | 0.833 | 0.392 | 0.4   | 0.371882561883  |
| k = 8 | Naive Bayes Classifier (without Tuning)         | 0.837 | 0.402 | 0.4   | 0.379847929848  |
| k = 9 | Naive Bayes Classifier (without Tuning)         | 0.836 | 0.323 | 0.324 | 0.309473304473  |

## Question 4:
> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune?

**Hyperparameter tuning** selects a set of optimal hyperparameters for machine learning algorithms. It can help to avoid overfitting and increase the performance of the algorithms on an independent dataset: 

- In case of the K nearest neighbor classifier, GridSearchCV searches through different combination of algorithms and n_neighbors.
- In case of SVC, GridSearchCV tries different value for C and gamma to avoid overfitting.
- In case of decision tree classifier, the GridSearchCV can try different value for criterion. Moreover, overfitting can be avoided for example by searching through different min_sample_split.
- In case of logistic regression classifier, trying different value for the parameter C can help can avoid overfitting as well.

**GridSearchCV** is deployed. As scoring method, the recall is used to decide the best parameters. 
1) For K nearest neighbor the parameter n_neighbors and algorithm are tunned:
```python
k_range = list(range(1,11))
algorithm_options = ['ball_tree','kd_tree','brute','auto']
param_grid_knn = dict(n_neighbors=k_range, algorithm=algorithm_options)
```
2) For support vector classifier, the parameter C, gamma and kernel are tunned:
```python
param_grid_svc = [
  {'C': [1, 10, 50, 100, 150, 1000], 'kernel': ['linear','rbf']},
  {'C': [1, 10, 50, 100, 150, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear','rbf']}]
```
 
3) For decision tree classifier, the parameter criterion, min_samples_split, max_depth, min_samples_leaf, and max_leaf_nodes are tunned:
```python
param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 10, 20],
    "max_depth": [None, 2, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    "max_leaf_nodes": [None, 5, 10, 20],
}
```
4) For logistic regression classifier, the parameter C is tunned:
```python
param_grid_l = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
```

After parameter tuning, we can see that the performance of decision tree classifier is improved: 

| K value | Classifier | Mean Accuracy | Mean of precision | Mean of recall | Mean of f1 score |
| ------- | ---------- | -------------:| -----------------:| --------------:| -----------:|
| k = 4 | Decision Tree Classifier | 0.775 | 0.18 | 0.22 | 0.192070707071 |
| k = 4 | Decision Tree Classifier Tune | 0.787 | 0.198 | 0.24 | 0.214203574204 |
| k = 5 | Decision Tree Classifier | 0.779 | 0.244 | 0.36 | 0.288171828172 | 
| k = 5 | Decision Tree Classifier Tune | 0.779 | 0.259 | 0.42 | 0.317517482517 |
| k = 8 | Decision Tree Classifier | 0.792 | 0.224 | 0.26 | 0.225356277709 | 
| k = 8 | Decision Tree Classifier Tune | 0.797 | 0.255 | 0.3 | 0.259738562092 |

## Question 5: 
> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  
 
In machine learning, validation is a method to test the model’s performance by splitting the data into training and testing data: train the model with the training dataset and test it with the testing dataset. The performance of the algorithm will be validated with the performance when the model is applied on the testing dataset. With validation the problem of overfitting can be avoided and the model’s performance can be improved when it is applied with an independent dataset. Without proper validation of the machine learning algorithm, overfitting can happened: the model can be overfitted with every single data point. In this way the model could have very high accuracy_score on the training data. However, the model will be failed in new cases / independent dataset, as the model doesn’t generalize the cases but simply ‘remember’ each single case.
 
This project applies the cross validation strategy: The dataset will be split into 10 folds, and each fold will be used both for testing as well as training. The performance of the model will be measured with the average recall_score. The cross validation strategy will have more advantage than simply splitting the data into training and testing sets in the enron dataset case. The main reason is that the data points in the Enron dataset is in total 145 (after removing the ‘TOTAL’ line) and with cross validation the training and testing dataset can be better splitted to avoid unequal splitting of across the class POI / non POI.

## Question 6: 
> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.
 
The project uses the following evaluation metrics: accuracy_score, precision_score, recall_score, and f1_score. When we take a closer look at the accuracy_score, most of them are above 0.8, which doesn’t tell much about the performance of model. Moreover, due to that the there is in total 18 person of interest out of 145 in the dataset, the accuracy_score will not be a strong index for the performance of model.
 
As the goal of the project is to identify the person of interest, namely how many person of interest are identified, the recall_score will be a better metrics here: 

```python
recall_score = number_correct_identified_POI / total_number_POI
```

If the model predicts every person as POI, then the recall_score will be 1. Therefore, the precision_score will be also looked at. The precision_score tells that how many predicted POI are true POI:

```python
precision_score = number_correct_identified_POI / total_number_POI_predicted
```

F1_score is the harmonic mean of precision and recall. 

To sum up the evaluation strategy:

1. Different K value are tried with **SelectKBest**
1. Algorithms with precision_score, recall_score and f1_score greater than 0.3 are selected.
1. The higher the recall is, the better the performance of an algorithms is.

With the 4 best features selected, the naive bayes shows the best performance on average:

1. Mean of accuracy = 0.861: 86.1% of the 145 data points are correctly predicted
1. Mean of precision: 0.511: Among the identified POI, 51.1% are true POI.
1. Mean of recall: 0.42: Among the 18 POI, 42% are correctly identified.
1. Mean of f1 score: 0.42847041847: The harmonic mean of precision and recall is 0.42.

## Sources
- http://machinelearningmastery.com/feature-selection-machine-learning-python/
- http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
- http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- http://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
- http://www.bigdataexaminer.com/2016/03/01/k-nearest-neighbors-and-curse-of-dimensionality-in-python-scikit-learn/
- http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
- http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
- https://en.wikipedia.org/wiki/Precision_and_recall
- https://en.wikipedia.org/wiki/F1_score


