import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('internship_train.csv')


features = ['6', '7']

X = np.array(df[features])
y = np.array(df['target'])

poly = PolynomialFeatures(2, include_bias=False)
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


df_test = pd.read_csv('internship_hidden_test.csv')


X_hidden_test = np.array(df_test[features])
X_hidden_test = poly.fit_transform(X_hidden_test)


poly_hidden_test_prediction = poly_regressor.predict(X_hidden_test)
poly_hidden_test_prediction = pd.DataFrame(
    poly_hidden_test_prediction,
    columns=['target'])

poly_hidden_test_prediction.to_csv('task3_prediction.csv')


