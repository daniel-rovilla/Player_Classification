#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:39:42 2020

@author: chascream
"""
# %% Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

os.chdir('/Users/chascream/Documents/Python Projects/Fifa')
df = pd.read_csv('players_20.csv')


df = df[df['value_eur'] > 1]
clubs = df.groupby(['club']).mean()
clubs = clubs[['overall', 'value_eur', 'wage_eur']].reset_index()

clubs_overall = clubs.sort_values('overall', ascending = False).reset_index()
f, ax = plt.subplots(figsize = (20,5))
sns.barplot(x = 'club', y = 'overall', data = clubs_overall.iloc[:10], palette="CMRmap")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set(ylim = (70,85))
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

clubs_value_eur = clubs.sort_values('value_eur', ascending = False).reset_index()
f, ax = plt.subplots(figsize = (20,5))
sns.barplot(x = 'club', y = 'value_eur', data = clubs_value_eur.iloc[:10], palette="CMRmap")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

clubs_wage_eur = clubs.sort_values('wage_eur', ascending = False).reset_index()
f, ax = plt.subplots(figsize = (20,5))
sns.barplot(x = 'club', y = 'wage_eur', data = clubs_wage_eur.iloc[:10], palette="CMRmap")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

# Most expensive player
most_expns_plyr = df.loc[df.value_eur == df['value_eur'].max(),'short_name']
print('The most expensive player is ', most_expns_plyr.values[0])

# Highest wage for a player
top_wage_plyr = df.loc[df.wage_eur == df['wage_eur'].max(),'short_name']
print('The player with the highest wage is ', top_wage_plyr.values[0])

# Most frequent jersey number
numbers = df['team_jersey_number'].value_counts()

# Most frequent Nationality
df['nationality'].value_counts().idxmax()


def combine_positions(row):
    # There are 15 different positions
    positions = row['player_positions'].split(', ') 
    N = len(positions)
    if N < 3:
        # If a player has two positions the first one will be considered as their
        # positions, of course for players with only one position won't be affected
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
# df['player_positions'] = df['player_positions'].replace([0,1,2,3],['ATT','MID', 'DEF', 'GK'])



bins = np.linspace(df.overall.min(), df.overall.max(), 10)
g = sns.FacetGrid(df, col="player_positions", hue="preferred_foot",
                  hue_order=['Right','Left'], palette="Set1", col_wrap=4)
g.map(plt.hist, 'overall', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


sns.distplot(df['overall'], hist=True, 
             bins=bins, color = 'blue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
print('Skew of the distribution is', df['overall'].skew())
print('Kurtosis of the distribution is', df['overall'].kurt())


