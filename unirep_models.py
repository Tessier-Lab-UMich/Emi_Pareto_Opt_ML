# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:13:21 2021

@author: makow
"""

from holdout_utils import *

emi_binding = pd.read_csv("emi_binding.csv", header = 0, index_col = 0)
iso_binding = pd.read_csv("iso_binding.csv", header = 0, index_col = 0)
igg_binding = pd.read_csv("igg_binding.csv", header = 0, index_col = 0)

emi_reps = pd.read_csv("emi_reps.csv", header = 0, index_col = 0)
iso_reps = pd.read_csv("iso_reps.csv", header = 0, index_col = 0)
igg_reps = pd.read_csv("igg_reps.csv", header = 0, index_col = 0)

#%%
lda_ant = LDA()
cv_results = cv(lda_ant, emi_reps, emi_binding.iloc[:,0])
print('Antigen model cross validation average test accuracy: ' + str(np.mean(cv_results['test_score'])))
emi_ant_transform = pd.DataFrame(lda_ant.fit_transform(emi_reps, emi_binding.iloc[:,0])).set_index(emi_binding.index)
emi_ant_predict = pd.DataFrame(lda_ant.predict(emi_reps)).set_index(emi_binding.index)
print('Antigen model accuracy: ' + str(accuracy_score(emi_ant_predict.iloc[:,0], emi_binding.iloc[:,0])))
iso_ant_transform = pd.DataFrame(lda_ant.transform(iso_reps)).set_index(iso_binding.index)
iso_ant_predict = pd.DataFrame(lda_ant.predict(iso_reps)).set_index(iso_binding.index)
igg_ant_transform = pd.DataFrame(lda_ant.transform(igg_reps)).set_index(igg_binding.index)

lda_psy = LDA()
cv_results = cv(lda_psy, emi_reps, emi_binding.iloc[:,1])
print('Specificity model cross validation average test accuracy: ' + str(np.mean(cv_results['test_score'])))
emi_psy_transform = pd.DataFrame(lda_psy.fit_transform(emi_reps, emi_binding.iloc[:,1])).set_index(emi_binding.index)
emi_psy_predict = pd.DataFrame(lda_psy.predict(emi_reps)).set_index(emi_binding.index)
print('Specificity model accuracy: ' + str(accuracy_score(emi_psy_predict.iloc[:,0], emi_binding.iloc[:,1])))
iso_psy_transform = pd.DataFrame(lda_psy.transform(iso_reps)).set_index(iso_binding.index)
iso_psy_predict = pd.DataFrame(lda_psy.predict(iso_reps)).set_index(iso_binding.index)
igg_psy_transform = pd.DataFrame(lda_psy.transform(igg_reps)).set_index(igg_binding.index)

#%%
"""
# sample size elbow plot
emi_data = pd.concat([emi_binding, emi_reps.set_index(emi_binding.index)], axis = 1)
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
plt.scatter(np.arange(1,25), ant_predict_acc, c = 'red', edgecolor = 'k', linewidth = 0.25, s = 50)
plt.scatter(np.arange(1,25), psy_predict_acc, c = 'blue', edgecolor = 'k', linewidth = 0.25, s = 50)
plt.xticks(fontsize = 24)
plt.yticks([0.8, 0.9, 1.0], [80, 90, 100], fontsize = 24)
"""
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
plt.scatter(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2], c = iso_psy_predict.iloc[:,0], cmap = cmap9, s = 150, edgecolor = 'k', linewidth = 0.25)
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

#print(len(emi_ant_transform))

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
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
#experimental pareto
plt.figure()
plt.errorbar(igg_binding.iloc[0:41,1], igg_binding.iloc[0:41,2], xerr = igg_binding.iloc[0:41,3], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_binding.iloc[0:41,1], igg_binding.iloc[0:41,2], s = 150, c = 'blueviolet', edgecolor = 'k', linewidth = 0.5, zorder = 2)
#plt.scatter(igg_binding.loc[igg_binding['Scaffold'] == 1,'ANT Binding'], igg_binding.loc[igg_binding['Scaffold'] == 1,'OVA Binding'], s = 150, c = cmap(0.65), edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.scatter(1,1, s = 200, c = 'k', edgecolor = 'k', linewidth = 0.5, zorder = 4)
plt.scatter(1.2,0.51, s = 200, c = cmap(0.85), edgecolor = 'k', linewidth = 0.5, zorder = 4)
plt.xticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.xlim(-0.05, 1.45)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)

#%%
#novel IgG correlations
plt.figure()
plt.errorbar(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3,4], [1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)
print('Antigen model novel IgG correlation: ' + str(sc.stats.spearmanr(igg_ant_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'])))


plt.figure()
plt.errorbar(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print('Specificity model novel IgG correlation: ' + str(sc.stats.spearmanr(igg_psy_transform.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'])))


#%%
#novel IgG correlations without Blosum62 filter
print('Antigen model novel IgG correlation: ' + str(sc.stats.spearmanr(igg_ant_transform.iloc[42:100,0], igg_binding.iloc[42:100,1])))
print('Specificity model novel IgG correlation: ' + str(sc.stats.spearmanr(igg_psy_transform.iloc[42:100,0], igg_binding.iloc[42:100,2])))


#%%
#experimental pareto
plt.figure()
plt.errorbar(igg_binding.iloc[0:41,1], igg_binding.iloc[0:41,2], xerr = igg_binding.iloc[0:41,3], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_binding.iloc[0:41,1], igg_binding.iloc[0:41,2], s = 150, c = 'blueviolet', edgecolor = 'k', linewidth = 0.5, zorder = 2)
plt.errorbar(igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], c = 'mediumspringgreen', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
#plt.scatter(igg_binding.loc[igg_binding['Scaffold'] == 1,'ANT Binding'], igg_binding.loc[igg_binding['Scaffold'] == 1,'OVA Binding'], s = 150, c = cmap(0.65), edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.scatter(1,1, s = 250, c = 'k', edgecolor = 'k', linewidth = 0.5, zorder = 4)
plt.scatter(1.2,0.51, s = 250, c = 'orange', edgecolor = 'k', linewidth = 0.5, zorder = 4)
plt.scatter(1.28, 0.3, s = 250, c = 'red', edgecolor = 'k', linewidth = 0.5, zorder = 4)

plt.xticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.xlim(-0.05, 1.45)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)



