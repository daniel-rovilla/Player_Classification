#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:16:53 2020

@author: chascream
"""
# %% Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
# %%

# Regression model between the Overall skills of a player with the value of the player


os.chdir('/Users/chascream/Documents/Python Projects/Fifa')
df = pd.read_csv('players_20.csv')
df = df[df.value_eur != 0]

X = df['overall']
y = df['value_eur']

import seaborn as sns
sns.regplot(X, y)

# Linear regression with plot
lin_reg = LinearRegression()
lin_reg.fit(np.array(X).reshape(-1, 1), y)
y_pred_lr = lin_reg.predict(np.array(X).reshape(-1,1))
plt.figure(figsize=(10,8));
plt.scatter(X, y, color='blue');
plt.plot(X, y_pred_lr, color='red');
print(r2_score(y, y_pred_lr))
# Same regression with more details
x, y = np.array(X), np.array(y)
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
results.summary()


# Cuadratic regression with plot
poly_reg_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_reg_2.fit_transform(np.array(X).reshape(-1, 1))
lin_reg.fit(X_poly_2, np.array(y).reshape(-1, 1))
y_pred_pr2 = lin_reg.predict(X_poly_2)
plt.figure(figsize=(10,8));
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_pr2, color='red')
print(r2_score(y, y_pred_pr2))
# Same regression with more details
x_p2, y_p2 = np.array(X_poly_2), np.array(y)
x_p2 = sm.add_constant(X_poly_2)
model_p2 = sm.OLS(y_p2, x_p2)
results_p2 = model_p2.fit()
results_p2.summary()


# Cubic regression with plot
poly_reg_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_reg_3.fit_transform(np.array(X).reshape(-1, 1))
lin_reg.fit(X_poly_3, np.array(y).reshape(-1, 1))
y_pred_pr3 = lin_reg.predict(X_poly_3)
plt.figure(figsize=(10,8));
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_pr3, color='red')
plt.title("3rd degree polynomial regression")
plt.xlabel("Overall")
plt.ylabel("Value")
plt.show()
print(r2_score(y, y_pred_pr3))
# Same regression with more details
x_p3, y_p3 = np.array(X_poly_3), np.array(y)
x_p3 = sm.add_constant(X_poly_3)
model_p3 = sm.OLS(y_p3, x_p3)
results_p3 = model_p3.fit()
results_p3.summary()

poly_reg_4 = PolynomialFeatures(degree=4)
X_poly_4 = poly_reg_4.fit_transform(np.array(X).reshape(-1, 1))
lin_reg.fit(X_poly_4, np.array(y).reshape(-1, 1))
y_pred_pr4 = lin_reg.predict(X_poly_4)
plt.figure(figsize=(10,8));
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_pr4, color='red')
plt.title("4rd degree polynomial regression")
plt.xlabel("Overall")
plt.ylabel("Value")
plt.show()
print(r2_score(y, y_pred_pr4))
# Same regression with more details
x_p4, y_p4 = np.array(X_poly_4), np.array(y)
x_p4 = sm.add_constant(X_poly_4)
model_p4 = sm.OLS(y_p4, x_p4)
results_p4 = model_p4.fit()
results_p4.summary()


# Testing the quatric regression model
X_train,X_test,y_train,y_test = train_test_split(X_poly_4, y, test_size=0.2,random_state=42)
lin_reg.fit(X_train, np.array(y_train).reshape(-1, 1))
model_pred = lin_reg.predict(X_train)
print('Accuracy: ', lin_reg.score(X_test, y_test))




