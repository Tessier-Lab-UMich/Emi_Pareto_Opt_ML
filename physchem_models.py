# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:13:21 2021

@author: makow
"""

from holdout_utils import *
from physchem_gen import physchemvh_gen

emi_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\Emi_Pareto_opt_ML\\emi_binding.csv", header = 0, index_col = 0)
iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\Emi_Pareto_opt_ML\\iso_binding.csv", header = 0, index_col = 0)
igg_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\Emi_Pareto_opt_ML\\igg_binding.csv", header = 0, index_col = 0)

emi_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\Emi_Pareto_opt_ML\\emi_pI.txt", sep = '\t', header = None, index_col = None)
iso_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\Emi_Pareto_opt_ML\\iso_pI.txt", sep = '\t',header = None, index_col = None)
igg_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\Emi_Pareto_opt_ML\\igg_pI.txt", sep = '\t',header = None, index_col = None)

#%%
emi_physvh = physchemvh_gen(emi_binding, emi_pI.iloc[:,1])
iso_physvh = physchemvh_gen(iso_binding, iso_pI.iloc[:,1])
igg_physvh = physchemvh_gen(igg_binding, igg_pI.iloc[:,1])

#%%
lda_ant = LDA()
cv_results = cv(lda_ant, emi_physvh, emi_binding.iloc[:,0])
emi_ant_transform = pd.DataFrame(lda_ant.fit_transform(emi_physvh, emi_binding.iloc[:,0])).set_index(emi_binding.index)
emi_ant_predict = pd.DataFrame(lda_ant.predict(emi_physvh)).set_index(emi_binding.index)
print(accuracy_score(emi_ant_predict.iloc[:,0], emi_binding.iloc[:,0]))
iso_ant_transform = pd.DataFrame(lda_ant.transform(iso_physvh)).set_index(iso_binding.index)
iso_ant_predict = pd.DataFrame(lda_ant.predict(iso_physvh)).set_index(iso_binding.index)
igg_ant_transform = pd.DataFrame(lda_ant.transform(igg_physvh)).set_index(igg_binding.index)

lda_psy = LDA()
cv_results = cv(lda_psy, emi_physvh, emi_binding.iloc[:,1])
emi_psy_transform = pd.DataFrame(-1*lda_psy.fit_transform(emi_physvh, emi_binding.iloc[:,1])).set_index(emi_binding.index)
emi_psy_predict = pd.DataFrame(lda_psy.predict(emi_physvh)).set_index(emi_binding.index)
print(accuracy_score(emi_psy_predict.iloc[:,0], emi_binding.iloc[:,1]))
iso_psy_transform = pd.DataFrame(-1*lda_psy.transform(iso_physvh)).set_index(iso_binding.index)
iso_psy_predict = pd.DataFrame(lda_psy.predict(iso_physvh)).set_index(iso_binding.index)
igg_psy_transform = pd.DataFrame(-1*lda_psy.transform(igg_physvh)).set_index(igg_binding.index)

#%%
plt.figure()
sns.distplot(emi_ant_transform.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(emi_ant_transform.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
plt.ylabel('')
plt.xlim(-5,5)

plt.figure()
sns.distplot(emi_psy_transform.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(emi_psy_transform.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1], c = iso_ant_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_ant_transform.iloc[125,0], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)
print(sc.stats.spearmanr(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1]))

plt.figure()
plt.scatter(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2], c = iso_psy_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_psy_transform.iloc[125,0], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)
print(sc.stats.spearmanr(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2]))

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[0:41,0], igg_psy_transform.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[8,0], igg_psy_transform.iloc[8,0], c = 'orange', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.errorbar(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3], [1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)
print(sc.stats.spearmanr(igg_ant_transform.iloc[0:42,0], igg_binding.iloc[0:42,1]))

plt.figure()
plt.errorbar(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print(sc.stats.spearmanr(igg_psy_transform.iloc[0:42,0], igg_binding.iloc[0:42,2]))

plt.figure()
plt.errorbar(igg_binding.iloc[0:41,1], igg_binding.iloc[0:41,2], xerr = igg_binding.iloc[0:41,3], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_binding.iloc[0:41,1], igg_binding.iloc[0:41,2], s = 150, c = cmap(0.15), edgecolor = 'k', linewidth = 0.5, zorder = 2)
plt.scatter(igg_binding.loc[igg_binding['Scaffold'] == 1,'ANT Binding'], igg_binding.loc[igg_binding['Scaffold'] == 1,'OVA Binding'], s = 150, c = cmap(0.65), edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.scatter(1,1, s = 200, c = 'k', edgecolor = 'k', linewidth = 0.5, zorder = 4)
plt.scatter(1.2,0.51, s = 200, c = cmap(0.85), edgecolor = 'k', linewidth = 0.5, zorder = 4)
plt.xticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.xlim(-0.05, 1.45)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)

plt.figure()
plt.errorbar(igg_ant_transform.iloc[42:103,0], igg_binding.iloc[42:103,1], yerr = igg_binding.iloc[42:103,3], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.iloc[42:103,0], igg_binding.iloc[42:103,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[8,0], igg_binding.iloc[8,1], c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3,4], [1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)

plt.figure()
plt.errorbar(igg_psy_transform.iloc[42:103,0], igg_binding.iloc[42:103,2], yerr = igg_binding.iloc[42:103,4], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.iloc[42:103,0], igg_binding.iloc[42:103,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[8,0], igg_binding.iloc[8,2], c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)

plt.figure()
plt.errorbar(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0,1, 2, 3,4], [0,1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)
print(sc.stats.spearmanr(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding']))

plt.figure()
plt.errorbar(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0,1, 2, 3,4,5], [0,1, 2, 3,4,5], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print(sc.stats.spearmanr(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding']))


