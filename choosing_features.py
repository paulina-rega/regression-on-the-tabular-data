'''
Program helping to determine polynomial degree and feature set for regression.

Prints information about root of mean squared error, R2 score and
execution time of regression for different parameters
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time


df = pd.read_csv('internship_train.csv')

def r2_rmse_scores(df, features, degree):
    '''
    Trains models and returns root of mean squared errror and R2 score for 
    regression model train and test datas of given polynomial degree 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'target' outcome
    features : list
        list of feature of DataFrame to use in regression
    degree : int
        degree of polynomial

    Returns
    -------
    rmse_test : float64
        Root of mean squared error for test data.
    r2_test : float64
        R2 score for test data.
    rmse_train : float64
        Root of mean squared error for train data.
    r2_train : float64
        R2 score for trains data.

    '''
    X = np.array(df[features])
    y = np.array(df['target'])

    poly = PolynomialFeatures(degree, include_bias=False)
    X = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=34)

    poly_regressor = LinearRegression(fit_intercept=False)
    poly_regressor.fit(X_train, y_train)

    poly_y_prediction = poly_regressor.predict(X_test)
    poly_y_prediction_train = poly_regressor.predict(X_train)
    
    rmse_test = np.sqrt(mean_squared_error(y_test, poly_y_prediction))
    r2_test = r2_score(y_test, poly_y_prediction)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, poly_y_prediction_train))
    r2_train = r2_score(y_train, poly_y_prediction_train)
    
    return (rmse_test, r2_test, rmse_train, r2_train)




def train_model_print_info(df, features, deg):
    '''
    Prints information about execution time, root of mean squared errror and 
    R2 score for regression model for train and test datas 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and 'target' outcome
    features : list
        list of feature of DataFrame to use in regression
    degree : int
        degree of polynomial


    '''
    
    start_time = time.time()
    mse = r2_rmse_scores(df, features, deg)
    print('-------DEGREE {}------'.format(deg))
    print('-'*10)
    print('For features: ', features)
    print('– TEST SET\n \tRoot of mean squared error: {} \n '\
          '\tR2 score: {} \n'\
          '– TRAIN SET\n \tRoot of mean squarre error: {}\n'\
          '\tR2 score: {}'.format(mse[0], mse[1], mse[2], mse[3]))
    print('– Execution time: ', (time.time() - start_time))



deg = 2
features_to_keep=df.columns.to_list()[:-1]
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7', '40', '26']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7', '40']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6']
train_model_print_info(df, features_to_keep, deg)


deg = 1
features_to_keep=df.columns.to_list()[:-1]
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7', '40', '26']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7', '40']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6']
train_model_print_info(df, features_to_keep, deg)


deg = 3

features_to_keep = ['6', '7', '40', '26']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7', '40']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6', '7']
train_model_print_info(df, features_to_keep, deg)

features_to_keep = ['6']
train_model_print_info(df, features_to_keep, deg)


# From the printed informations the optimal outcome is for
# 2nd degree polynomial regression model trained with features '6' and '7'
