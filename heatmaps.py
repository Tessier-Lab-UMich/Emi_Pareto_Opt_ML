# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:37:01 2021

@author: makow
"""

from holdout_utils import *

#%%
figure4 = pd.read_csv("figure4_heatmap.csv", header = 0, index_col = 0)
plt.figure(figsize = (10,4))
sns.heatmap(figure4.iloc[0:3,0:3].T, annot = figure4.iloc[0:3,3:6].T, fmt = '', annot_kws = {'fontsize': 26, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 1.05)
plt.rcParams["font.family"] = 'Myriad Pro'

plt.figure(figsize = (10,10))
sns.heatmap(figure4.iloc[0:3,6:9].T, annot = figure4.iloc[0:3,9:12].T, fmt = '', annot_kws = {'fontsize': 26, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 1.05)
plt.rcParams["font.family"] = 'Myriad Pro'

