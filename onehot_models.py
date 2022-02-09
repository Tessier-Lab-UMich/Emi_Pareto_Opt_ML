# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:13:21 2021

@author: makow
"""

from holdout_utils import *
from onehot_gen import onehot_gen, shiftedColorMap
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns

emi_binding = pd.read_csv("emi_binding.csv", header = 0, index_col = 0)
iso_binding = pd.read_csv("iso_binding.csv", header = 0, index_col = 0)
igg_binding = pd.read_csv("igg_binding.csv", header = 0, index_col = 0)

#%%
emi_onehot = onehot_gen(emi_binding)
iso_onehot = onehot_gen(iso_binding)
igg_onehot = onehot_gen(igg_binding)

#%%
lda_ant = LDA()
cv_results = cv(lda_ant, emi_onehot, emi_binding.iloc[:,0])
print('Antigen model cross validation average test accuracy: ' + str(np.mean(cv_results['test_score'])))
emi_ant_transform = pd.DataFrame(lda_ant.fit_transform(emi_onehot, emi_binding.iloc[:,0])).set_index(emi_binding.index)
emi_ant_predict = pd.DataFrame(lda_ant.predict(emi_onehot)).set_index(emi_binding.index)
print('Antigen model accuracy: ' + str(accuracy_score(emi_ant_predict.iloc[:,0], emi_binding.iloc[:,0])))
iso_ant_transform = pd.DataFrame(lda_ant.transform(iso_onehot)).set_index(iso_binding.index)
iso_ant_predict = pd.DataFrame(lda_ant.predict(iso_onehot)).set_index(iso_binding.index)
igg_ant_transform = pd.DataFrame(lda_ant.transform(igg_onehot)).set_index(igg_binding.index)

lda_psy = LDA()
cv_results = cv(lda_psy, emi_onehot, emi_binding.iloc[:,1])
print('Specificity model cross validation average test accuracy: ' + str(np.mean(cv_results['test_score'])))
emi_psy_transform = pd.DataFrame(lda_psy.fit_transform(emi_onehot, emi_binding.iloc[:,1])).set_index(emi_binding.index)
emi_psy_predict = pd.DataFrame(lda_psy.predict(emi_onehot)).set_index(emi_binding.index)
print('Specificity model accuracy: ' + str(accuracy_score(emi_psy_predict.iloc[:,0], emi_binding.iloc[:,1])))
iso_psy_transform = pd.DataFrame(lda_psy.transform(iso_onehot)).set_index(iso_binding.index)
iso_psy_predict = pd.DataFrame(lda_psy.predict(iso_onehot)).set_index(iso_binding.index)
igg_psy_transform = pd.DataFrame(lda_psy.transform(igg_onehot)).set_index(igg_binding.index)

#%%
# sample size elbow plot
emi_data = pd.concat([emi_binding, emi_onehot.set_index(emi_binding.index)], axis = 1)
ant_test_acc = []
psy_test_acc = []
for i in np.arange(25,4000,25):
    emi_data_subset = emi_data.sample(i)
    emi_data_subset_train, emi_data_subset_test, emi_data_subset_target_train, emi_data_subset_target_test = train_test_split(emi_data_subset.iloc[:,3:8000], emi_data_subset.iloc[:,0:3])
    cv_results = cv(lda_ant, emi_data_subset.iloc[:,3:8000], emi_data_subset.iloc[:,0])
    ant_test_acc.append(np.mean(cv_results['test_score']))
    cv_results = cv(lda_psy, emi_data_subset.iloc[:,3:8000], emi_data_subset.iloc[:,1])
    psy_test_acc.append(np.mean(cv_results['test_score']))

#%%
plt.scatter(np.arange(25,4000,25), ant_test_acc, c = 'blue', edgecolor = 'k', linewidth = 0.25, s = 50)
plt.scatter(np.arange(25,4000,25), psy_test_acc, c = 'red', edgecolor = 'k', linewidth = 0.25, s = 50)
plt.xticks([0,1000,2000,3000,4000], fontsize = 24)
plt.yticks([0.5, 0.6, 0.7,0.8, 0.9, 1.0], [50, 60, 70, 80, 90, 100], fontsize = 24)

#%%
#KNN of sequences
from sklearn.neighbors import KNeighborsClassifier as KNC

ant_predict_acc = []
psy_predict_acc = []
for j in np.arange(1,25):
    knc = KNC(n_neighbors = j)
    cv_results = cv(knc, emi_data.iloc[:,3:8000], emi_data.iloc[:,0])
    ant_predict_acc.append(np.mean(cv_results['test_score']))
    cv_results = cv(knc, emi_data.iloc[:,3:8000], emi_data.iloc[:,1])
    psy_predict_acc.append(np.mean(cv_results['test_score']))
    
#%%
plt.scatter(np.arange(1,25), ant_predict_acc, c = 'blue', edgecolor = 'k', linewidth = 0.25, s = 50)
plt.scatter(np.arange(1,25), psy_predict_acc, c = 'red', edgecolor = 'k', linewidth = 0.25, s = 50)
plt.xticks(fontsize = 24)
plt.yticks([0.8, 0.9, 1.0], [80, 90, 100], fontsize = 24)

#%%
#model accuracy distributions
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

#%%
#yeast data correlations
plt.figure()
plt.scatter(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1], c = iso_ant_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_ant_transform.iloc[125,0], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)
print('Antigen model scFab correlation: ' + str(sc.stats.spearmanr(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1])))

plt.figure()
plt.scatter(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2], c = iso_psy_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_psy_transform.iloc[125,0], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)
print('Specificity model scFab correlation: ' + str(sc.stats.spearmanr(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2])))

#%%
#pareto plots
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
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation']==1),0], igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation']==1),0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[8,0], igg_psy_transform.iloc[8,0], c = 'orange', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation']==0),0], igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation']==0),0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[8,0], igg_psy_transform.iloc[8,0], c = 'orange', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')

#%%
#in-library IgG correlations
plt.figure()
plt.errorbar(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3], [1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)
print('Antigen model in-library IgG correlation: ' + str(sc.stats.spearmanr(igg_ant_transform.iloc[0:42,0], igg_binding.iloc[0:42,1])))

plt.figure()
plt.errorbar(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print('Specificity model in-library IgG correlation: ' + str(sc.stats.spearmanr(igg_psy_transform.iloc[0:42,0], igg_binding.iloc[0:42,2])))

#%%
#novel IgG correlations
plt.figure()
plt.errorbar(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'ANT Binding'], yerr = igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3,4], [1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)

plt.figure()
plt.errorbar(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'ANT Binding'], yerr = igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3,4], [1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)

plt.figure()
plt.errorbar(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3,4], [1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)

print('Antigen model novel IgG correlation, interpolation: ' + str(sc.stats.mstats.spearmanr(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'ANT Binding'], use_ties = True)))
print('Antigen model novel IgG correlation, extrapolation: ' + str(sc.stats.mstats.spearmanr(igg_ant_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'ANT Binding'], use_ties = True)))
print('Antigen model novel IgG correlation: ' + str(sc.stats.mstats.spearmanr(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], use_ties = True)))

plt.figure()
plt.errorbar(igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'OVA Binding'], yerr = igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'OVA STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-2, -1, 0,1, 2, 3], [-2, -1, 0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)

plt.figure()
plt.errorbar(igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'OVA Binding'], yerr = igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'OVA STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-2, -1, 0,1, 2, 3], [-2, -1, 0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)

plt.figure()
plt.errorbar(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-2, -1, 0,1, 2, 3], [-2, -1, 0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print('Specificity model novel IgG correlation, interpolation: ' + str(sc.stats.mstats.spearmanr(igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 1),'OVA Binding'], use_ties = True)))
print('Specificity model novel IgG correlation, extrapolation: ' + str(sc.stats.mstats.spearmanr(igg_psy_transform.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),0], igg_binding.loc[(igg_binding['Blosum62'] == 1) & (igg_binding['Interpolation'] == 0),'OVA Binding'], use_ties = True)))
print('Specificity model novel IgG correlation: ' + str(sc.stats.mstats.spearmanr(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], use_ties = True)))


#%%
#library site sequence analysis
vhLen = 115
siteLib = np.arange(1,vhLen+1,1)
libSites = np.array([32, 49, 54, 55, 56, 98, 100, 103])
AAlib = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
encodedSeqs = np.zeros((len(siteLib), 4000))
#Reshape lda coefficients to matrix for affinity
ldaScaleMatAff = np.zeros((len(siteLib), len(AAlib)))
c=0
for ii in range(len(siteLib)):
    for jj in range(len(AAlib)):
        ldaScaleMatAff[ii,jj] = lda_ant.coef_[0][c]
        c+=1

for ii in range(4000):
    encodedSeqs[:,ii] = le.transform(list(emi_binding.index[ii]))

uniqueLibMuts = np.array([0,2,3,4,5,8,9,11,14,15,16,17,18,19])
ldaScaleMatAffFilt = ldaScaleMatAff[libSites.astype(int), :]
ldaScaleMatAffFilt = ldaScaleMatAffFilt[:,uniqueLibMuts.astype(int)]


#Reshape lda coefficients for specificity
ldaScaleMatSpec = np.zeros((len(siteLib), len(AAlib)))
c=0
for ii in range(len(siteLib)):
    for jj in range(len(AAlib)):
        ldaScaleMatSpec[ii,jj] = lda_psy.coef_[0][c]
        c+=1
ldaScaleMatSpecFilt = ldaScaleMatSpec[libSites.astype(int), :]
ldaScaleMatSpecFilt = ldaScaleMatSpecFilt[:,uniqueLibMuts.astype(int)]

#Mask heatmap by actual library sites
wtResidues = ['Y', 'R','R','R','G','A','W','Y']
sampResidues = [['YFVASD'],
                ['RKGATE'],
                ['RKGATE'],
                ['RKGATE'],
                ['GANSTD'],
                ['AFVYSD'],
                ['WLVGAS'],
                ['YFVASD']]
scaleMask = np.zeros(ldaScaleMatAffFilt.shape)
uniqueLibMutsStr = np.array([AAlib[ii] for ii in uniqueLibMuts.astype(int)])
wtList = []
#Check if the residue is actually mutated for masking colormaps
for ii in range(len(sampResidues)):
    currWT = wtResidues[ii]
    currWTidx = np.where(uniqueLibMutsStr==currWT)[0][0]
    wtList.append((ii,currWTidx))
    for jj in range(len(sampResidues[ii][0])):
        currRes = sampResidues[ii][0][jj]
        #Find index of currRes in uniqueLibMutsStr
        currResIdx = np.where(uniqueLibMutsStr==currRes)
        scaleMask[ii,currResIdx] = 1

nonZeroRows = np.where(np.any(scaleMask!=0,axis =1))
nonZeroCols = np.where(np.any(scaleMask!=0,axis =0))
#Set unmutated regions to NaN for plotting
ldaScaleMatSpecFilt[~scaleMask.astype(bool)]=np.nan
ldaScaleMatAffFilt[~scaleMask.astype(bool)]=np.nan
yticklbls = AAlib[uniqueLibMuts[nonZeroCols].astype(int)]
xtickLbls = libSites
nanMask = np.isnan(ldaScaleMatSpecFilt)

#%%
######### Plot both colormaps
plt.figure()
plt.subplot(1,2,1)
ytickLbls = libSites
# nanMask = np.isnan(ldaScaleMatAffFilt)
ax=sns.heatmap(ldaScaleMatAffFilt.T, yticklabels=yticklbls, xticklabels=xtickLbls, square = True, cmap = "bwr", vmin = -3, vmax = 3, mask = nanMask.T)
ax.set_facecolor([0.85, 0.85, 0.85])

from matplotlib.patches import Rectangle

for ii in range(len(sampResidues)):
    
    ax.add_patch(Rectangle(wtList[ii], 1, 1, fill=False, edgecolor='black', lw=4))
# plt.show()
plt.title('Affinity')
plt.subplot(1,2,2)
ytickLbls = libSites
nanMask = np.isnan(ldaScaleMatSpecFilt)
ax=sns.heatmap(ldaScaleMatSpecFilt.T, yticklabels=yticklbls, xticklabels=xtickLbls, square = True, cmap = "bwr", vmin = -3, vmax = 3, mask = nanMask.T)
ax.set_facecolor([0.85, 0.85, 0.85])

from matplotlib.patches import Rectangle

for ii in range(len(sampResidues)):
    
    ax.add_patch(Rectangle(wtList[ii], 1, 1, fill=False, edgecolor='black', lw=4))
plt.show()
plt.title('Specificity')

#%%
emi_seq = pd.DataFrame(emi_binding.index)
emi_seq.columns = ['Sequence']
mutations = []
for i in emi_seq['Sequence']:
    characters = list(i)
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations = pd.DataFrame(mutations)
mutations.set_index(emi_binding.index, inplace = True)

sampResidues2 = ['YFVASD',
                'RKGATE',
                'RKGATE',
                'RKGATE',
                'GANSTD',
                'AFVYSD',
                'WLVGAS',
                'YFVASD']

#%%
import math
diff_agg_ant = pd.DataFrame()
for i in np.arange(8):
    diff = []
    for j in list(sampResidues2[i]):
        diff.append([j,math.log((len(mutations.loc[(mutations.iloc[:,i]==j)&(emi_binding.iloc[:,0]==1)])/1516)/(len(mutations.loc[(mutations.iloc[:,i]==j)&(emi_binding.iloc[:,0]==0)])/2484),2)])
    diff = pd.DataFrame(diff)
    diff.set_index(0, inplace = True)
    diff_agg_ant = pd.concat([diff_agg_ant, diff], axis = 1)
diff_agg_ant.sort_index(inplace = True)

diff_agg_spec = pd.DataFrame()
for i in np.arange(8):
    diff = []
    for j in list(sampResidues2[i]):
        diff.append([j,math.log((len(mutations.loc[(mutations.iloc[:,i]==j)&(emi_binding.iloc[:,1]==1)])/2000)/(len(mutations.loc[(mutations.iloc[:,i]==j)&(emi_binding.iloc[:,1]==0)])/2000),2)])
    diff = pd.DataFrame(diff)
    diff.set_index(0, inplace = True)
    diff_agg_spec = pd.concat([diff_agg_spec, diff], axis = 1)
diff_agg_spec.sort_index(inplace = True)

#%%
shifted_bwr = shiftedColorMap(plt.cm.bwr, midpoint=0.72, name='shifted')

libSites = np.array([32, 49, 54, 55, 56, 98, 100, 103])
ytickLbls = libSites
xticklbls = AAlib[uniqueLibMuts.astype(int)]
plt.figure()
plt.subplot(1,2,1)
ytickLbls = libSites
nanMask = np.isnan(diff_agg_ant)
ax=sns.heatmap(diff_agg_ant, yticklabels=xticklbls, xticklabels=ytickLbls, square = True, cmap = shifted_bwr, mask = nanMask, vmin = -5, vmax = 2)
ax.set_facecolor([0.85, 0.85, 0.85])

from matplotlib.patches import Rectangle

for ii in range(len(sampResidues2)):
    
    ax.add_patch(Rectangle(wtList[ii], 1, 1, fill=False, edgecolor='black', lw=4))
plt.show()

plt.subplot(1,2,2)
ytickLbls = libSites
nanMask = np.isnan(diff_agg_spec)
ax=sns.heatmap(diff_agg_spec, yticklabels=xticklbls, xticklabels=ytickLbls, square = True, cmap = "bwr",  mask = nanMask, vmin = -4, vmax = 4)
ax.set_facecolor([0.85, 0.85, 0.85])

from matplotlib.patches import Rectangle

for ii in range(len(sampResidues)):
    
    ax.add_patch(Rectangle(wtList[ii], 1, 1, fill=False, edgecolor='black', lw=4))
plt.show()

#%%
plt.figure()
plt.scatter(ldaScaleMatAffFilt.T, diff_agg_ant, s = 100, edgecolor = 'k', linewidth = 0.25, c = cmap(0.15))
plt.yticks([-6, -4, -2,  0, 2,  4], fontsize = 26)
plt.xticks(fontsize = 26)
print('Antigen model scalings and site enrichment correlation: ' + str(sc.stats.spearmanr(pd.DataFrame(ldaScaleMatAffFilt.T).values.flatten(), diff_agg_ant.values.flatten(), nan_policy = 'omit')))

plt.figure()
plt.scatter(ldaScaleMatSpecFilt.T, diff_agg_spec, s = 100, edgecolor = 'k', linewidth = 0.25, c = cmap(0.15))
plt.yticks([-4, -2, 0, 2, 4], fontsize = 26)
plt.xticks(fontsize = 26)
print('Specificity model scalings and site enrichment correlation: ' + str(sc.stats.spearmanr(pd.DataFrame(ldaScaleMatSpecFilt.T).values.flatten(), diff_agg_spec.values.flatten(), nan_policy = 'omit')))



