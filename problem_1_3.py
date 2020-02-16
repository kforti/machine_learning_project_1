from sklearn import linear_model
from sklearn import neighbors 
from sklearn import datasets

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np 

# Load Dataset 
diabetes_feature_data, diabetes_ground_truth_value = datasets.load_diabetes(True)

diabetes_feature_data_train, diabetes_feature_data_test, diabetes_ground_truth_value_train, diabetes_ground_truth_value_test = train_test_split(diabetes_feature_data, diabetes_ground_truth_value)

# Part 1 - Linear Regression
print("Linear Regression")
## Create and fit model 
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(diabetes_feature_data_train, diabetes_ground_truth_value_train);

## Predict the values 
linear_regression_predictions = linear_regression_model.predict(diabetes_feature_data_test);
## Calculate the error 
linear_regression_r2_score = mean_squared_error(diabetes_ground_truth_value_test, linear_regression_predictions)
print('MSE without CV: %.2f'% linear_regression_r2_score)

## Use cross validation to calcualte the mse 
scores = cross_val_score(linear_regression_model, diabetes_feature_data, diabetes_ground_truth_value, cv=10, scoring='neg_mean_squared_error')
print("MSE with CV: %0.2f" % (scores.mean() * -1))


# Part 2 - Ridge Regression
print("Ridge Regression")
## Create and fit model 
ridge_regression_model = linear_model.Ridge()
ridge_regression_model.fit(diabetes_feature_data_train, diabetes_ground_truth_value_train);

## Predict the values 
ridge_regression_predictions = ridge_regression_model.predict(diabetes_feature_data_test);
## Calculate the error 
ridge_regression_r2_score = mean_squared_error(diabetes_ground_truth_value_test, ridge_regression_predictions)
print('MSE without CV: %.2f'% ridge_regression_r2_score)

## Use cross validation to calcualte the mse 
scores = cross_val_score(ridge_regression_model, diabetes_feature_data, diabetes_ground_truth_value, cv=10, scoring='neg_mean_squared_error')
print("MSE with CV: %0.2f" % (scores.mean() * -1))


# Part 3 - Bayesian Ridge Regression 
print("Bayesian Ridge Regression")
## Create and fit model 
bayesian_ridge_regression_model = linear_model.BayesianRidge()
bayesian_ridge_regression_model.fit(diabetes_feature_data_train, diabetes_ground_truth_value_train);

## Predict the values 
bayesian_ridge_regression_predictions = ridge_regression_model.predict(diabetes_feature_data_test);
## Calculate the error 
bayesian_ridge_regression_r2_score = mean_squared_error(diabetes_ground_truth_value_test, bayesian_ridge_regression_predictions)
print('MSE without CV: %.2f'% bayesian_ridge_regression_r2_score)

## Use cross validation to calcualte the mse 
scores = cross_val_score(bayesian_ridge_regression_model, diabetes_feature_data, diabetes_ground_truth_value, cv=10, scoring='neg_mean_squared_error')
print("MSE with CV: %0.2f" % (scores.mean() * -1))

# Part 4 - KNN 
print("K-Nearest Neighbors")
## Create and fit model 
knn_model = neighbors.KNeighborsRegressor();
knn_model.fit(diabetes_feature_data_train, diabetes_ground_truth_value_train);

## Predict the values 
knn_predictions = knn_model.predict(diabetes_feature_data_test);
## Calculate the error 
knn_r2_score = mean_squared_error(diabetes_ground_truth_value_test, knn_predictions)
print('MSE without CV: %.2f'% knn_r2_score)

## Use cross validation to calcualte the mse 
scores = cross_val_score(knn_model, diabetes_feature_data, diabetes_ground_truth_value, cv=10, scoring='neg_mean_squared_error')
print("MSE with CV: %0.2f" % (scores.mean() * -1))
