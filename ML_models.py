#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:51 2020

@author: chascream
"""
# %% Libraries
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
# %%

os.chdir('/Users/chascream/Documents/Python Projects/Fifa')
df = pd.read_csv('players_20.csv')
df = df[df['value_eur'] > 1]
# Create a new dataframe with just the useful variables
df = df[['skill_moves', 'player_positions', 'attacking_crossing', 'attacking_finishing',
         'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
         'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
         'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 
         'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power',
         'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
         'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
         'mentality_vision', 'mentality_penalties', 'mentality_composure',
         'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',
         'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
         'goalkeeping_positioning', 'goalkeeping_reflexes']]


df.info()

# Create a new dataframe to look at some stats, visualization on some of them later
dfstats = df.describe()

# Find how many null values the data has
df.isnull().sum()

# Group players by positions
df.groupby('player_positions').agg(['count'])
''' Looking at this we can notice that some players have more than 1 position
    the first position will be considered as their main position  
'''

# player_positions = df['player_positions'].str.split(',')
df.player_positions[35:45]

def combine_positions(row):
    # There are 15 different positions
    positions = row['player_positions'].split(', ') 
    N = len(positions)
    if N < 3:
        # If a player has two positions the first one will be considered as their
        # position, of course, players with only one position won't be affected
        position = positions[0]
        if position in ['ST', 'LW', 'RW','CF']: #4
            return 0 #ATTACKER
        elif position in ['CAM', 'LM', 'CM', 'RM', 'CDM']: #5
            return 1 #MIDFIELDER
        elif position in ['LWB', 'RWB', 'LB', 'CB', 'RB']: #5
            return 2 #DEFENDER
        elif position in ['GK']: #1
            return 3 #GOALKEEPER
    else: # If player has three possible positions
        position_count = [0, 0, 0, 0] 
        for position in positions:
            if position in ['ST', 'LW', 'RW','CF']: #4
                index = 0 #ATTACKER
            elif position in ['CAM', 'LM', 'CM', 'RM', 'CDM']: #5
                index = 1 #MIDFIELDER
            elif position in ['LWB', 'RWB', 'LB', 'CB', 'RB']: #5
                index = 2 #DEFENDER
            elif position in ['GK']: #1
                index = 3 #GOALKEEPER
            else:
                continue 
            position_count[index] += 1 
        # This will count which was the most repeated position and assign it
        # to the player
        return position_count.index(max(position_count))

df['player_positions'] = df.apply(combine_positions, axis=1)

 # %% Pre-processing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create X and y
X = df.drop(["player_positions"],axis = 1)
y = df.player_positions

# Split the data to 80-20
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

# %% HeatMap
import seaborn as sns
import matplotlib.pyplot as plt

mask = np.zeros_like(X_train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

plt.figure(figsize=(10,10))
plt.title("Player Skills Correlation Matrix", fontsize=20)
x = sns.heatmap(
    X_train.corr(), 
    cmap='coolwarm',
    annot=True,
    fmt=".1f",
    mask=mask,
    linewidths = .5,
    vmin = -1, 
    vmax = 1,
)

# %% KNN
from sklearn.neighbors import KNeighborsClassifier

pipe_knn = Pipeline([
    ('sc', StandardScaler()),
    ('knn', KNeighborsClassifier())
    ])

params_knn = {
    'knn__n_neighbors': range(1, 20)
    }

search_knn = GridSearchCV(estimator=pipe_knn,
                      param_grid=params_knn,
                      cv = 5,
                      return_train_score=True)

search_knn.fit(X_train, y_train)
print(f" Best score is: {search_knn.best_score_} with parameters: {search_knn.best_params_}")


knn = KNeighborsClassifier(n_neighbors = 19)
# Train the model
knn.fit(X_train, y_train)
# Get the predict value from X_test
y_pred_knn = knn.predict(X_test)


from sklearn.metrics import classification_report
# Report
print(classification_report(y_pred_knn,y_test, digits = 3))
print('accuracy: ', metrics.accuracy_score(y_pred_knn, y_test))


from sklearn.metrics import plot_confusion_matrix
# Confusion Matrix
titles_options = [("Unnormalized Confusion Matrix", None),
                  ("Normalized Confusion Matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn.fit(X_train, y_train), X_test, y_test,
                                  display_labels=['ATT', 'MID', 'DEF', 'GK'],
                                  cmap='Blues',
                                  normalize=normalize)
    disp.ax_.set_title(title)

# %% Decision tree 
from sklearn.tree import DecisionTreeClassifier
import numpy as np

pipe_decsT = Pipeline([
    ('sc', StandardScaler()),
    ('decsT', DecisionTreeClassifier())
    ])

params_decsT = {
    'decsT__criterion' : ['gini', 'entropy'],
    'decsT__max_depth' : np.arange(3, 15)
    }

search_decsT = GridSearchCV(estimator=pipe_decsT,
                      param_grid=params_decsT,
                      cv = 5,
                      return_train_score=True)

search_decsT.fit(X_train, y_train)
print(f" Best score is: {search_decsT.best_score_} with parameters: {search_decsT.best_params_}")


scores_decsT = search_decsT.cv_results_['mean_test_score']
criterion = ['gini', 'entropy']
DEPTH = np.arange(3, 15)
for idx, criterion in enumerate(criterion):
    for score, depth in (zip(scores_decsT[idx*len(DEPTH): (idx+1)*len(DEPTH)], DEPTH)):
        print(f"{depth, criterion}: {score:.10f}")
        
        
decsT = DecisionTreeClassifier(max_depth=8)
decsT.fit(X_train,y_train)
y_pred_decsT = decsT.predict(X_test)


# Report
print(classification_report(y_pred_decsT,y_test, digits = 3))
print('accuracy: ', metrics.accuracy_score(y_pred_decsT,y_test))


# Confusion Matrix
for title, normalize in titles_options:
    disp = plot_confusion_matrix(decsT.fit(X_train, y_train), X_test, y_test,
                                  display_labels=['ATT', 'MID', 'DEF', 'GK'],
                                  cmap='Blues',
                                  normalize=normalize)
    disp.ax_.set_title(title)

# %% Support Vector machine
from sklearn.svm import SVC
import numpy as np

kernels = ['rbf', 'poly', 'sigmoid']
C = np.logspace(-2, 10, 13)

pipe_svm = Pipeline([
    ('sc', StandardScaler()),
    ('SVM', SVC())
    ])

params_svm = {'SVM__C': C,
              'SVM__kernel': kernels,
             }

search_svm = GridSearchCV(estimator=pipe_svm,
                      param_grid=params_svm,
                      cv = 5,
                      return_train_score=True)

search_svm.fit(X_train, y_train)
print(f" Best score is: {search_svm.best_score_} with parameters: {search_svm.best_params_}")


svc = SVC()
svc.fit(X_train,y_train)
y_pred_svm = svc.predict(X_test)


# Report
print(metrics.classification_report(y_pred_svm,y_test, digits = 3))
print('accuracy: ', metrics.accuracy_score(y_pred_svm,y_test))


# Confusion Matrix
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svc.fit(X_train, y_train), X_test, y_test,
                                  display_labels=['ATT', 'MID', 'DEF', 'GK'],
                                  cmap='Blues',
                                  normalize=normalize)
    disp.ax_.set_title(title)

# %% Logistic Regression
from sklearn.linear_model import LogisticRegression

solver = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
C = np.logspace(-2, 10, 13)

pipe_lr = Pipeline([
    ('sc', StandardScaler()),
    ('LR', LogisticRegression())
    ])

params_lr = {
    'LR__C': C,
    'LR__solver': solver
    }

search_lr = GridSearchCV(estimator=pipe_lr,
                      param_grid=params_lr,
                      cv = 5,
                      return_train_score=True)

search_lr.fit(X_train, y_train)
print(f" Best score is: {search_lr.best_score_} with parameters: {search_lr.best_params_}")


lr = LogisticRegression(C=10, solver='newton-cg') 
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)


# Report
print(metrics.classification_report(y_pred_lr,y_test, digits = 3))
print('accuracy: ', metrics.accuracy_score(y_pred_lr,y_test))


# Confusion Matrix
for title, normalize in titles_options:
    disp = plot_confusion_matrix(lr.fit(X_train, y_train), X_test, y_test,
                                  display_labels=['ATT', 'MID', 'DEF', 'GK'],
                                  cmap='Blues',
                                  normalize=normalize)
    disp.ax_.set_title(title)

# %% Metrics 
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# Evaluate KNN 
jc1 = (jaccard_score(y_test, y_pred_knn, average='weighted'))
fs1 = (f1_score(y_test, y_pred_knn,average='weighted'))

# Evaluate Decision Trees
jc2 = (jaccard_score(y_test, y_pred_decsT, average='weighted'))
fs2 = (f1_score(y_test, y_pred_decsT, average='weighted'))

# Evaluate SVM
jc3 = (jaccard_score(y_test, y_pred_svm, average='weighted'))
fs3 = (f1_score(y_test, y_pred_svm, average='weighted'))

# Evaluate Logistic Regression
jc4 = (jaccard_score(y_test, y_pred_lr, average='weighted'))
fs4 = (f1_score(y_test, y_pred_lr, average='weighted'))
LR_yhat_prob = lr.predict_proba(X_test)

list_jc = [jc1, jc2, jc3, jc4]
list_fs = [fs1, fs2, fs3, fs4]
list_ll = ['NA', 'NA', 'NA',(log_loss(y_test, LR_yhat_prob))]


# Fomulate the metrics-report
report = pd.DataFrame( list_jc,index=['KNN','Decision Tree','SVM','Logistic Regression'])
report.columns = ['Jaccard']
report.insert(loc=1, column='F1-score', value=list_fs)
report.insert(loc=2, column='LogLoss', value=list_ll)
report.columns.name = 'Algorithm'
report


# Time
knn_fit = search_knn.cv_results_['mean_fit_time'].sum()
decsT_fit = search_decsT.cv_results_['mean_fit_time'].sum()
svm_fit = search_svm.cv_results_['mean_fit_time'].sum()
lr_fit = search_lr.cv_results_['mean_fit_time'].sum()

knn_score = search_knn.cv_results_['mean_score_time'].sum()
decsT_score = search_decsT.cv_results_['mean_score_time'].sum()
svm_score = search_svm.cv_results_['mean_score_time'].sum()
lr_score = search_lr.cv_results_['mean_score_time'].sum()

time_knn = knn_fit + knn_score
time_decsT = decsT_fit + decsT_score
time_svm = svm_fit + svm_score
time_lr = lr_fit + lr_score

time_fit = [knn_fit, decsT_fit, svm_fit, lr_fit]
time_score = [knn_score, decsT_score, svm_score, lr_score]
time_fit = [time_knn, time_decsT, time_svm, time_lr]
total_time = [5*x for x in time_fit]

time_report = pd.DataFrame( time_fit,index=['KNN','Decision Tree','SVM','Logistic Regression'])
time_report.columns = ['Time for fit']
time_report.insert(loc=1, column='Time for score', value=time_score)
time_report.insert(loc=2, column='Time per k-fold', value=time_fit)
time_report.insert(loc=3, column='Total time', value=total_time)
time_report.columns.name = 'Algorithm'
time_report.style.format("{:.2f}")