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


iso_ho = ho_seq_ind(iso_binding.index)
iso_ho_reps = ho_reps(iso_ho, iso_reps)
iso_ho_physchemvh = ho_physchemvh(iso_ho, iso_binding)
iso_ho_binding = ho_binding(iso_ho, iso_binding)

emi_ho_inverse = ho_seq_ind_inverse(emi_binding.index)
emi_ho_reps_inverse = ho_reps(emi_ho_inverse, emi_reps)
emi_ho_physchemvh_inverse = ho_physchemvh(emi_ho_inverse, emi_binding)
emi_ho_binding_inverse = ho_binding(emi_ho_inverse, emi_binding)


#%%
lda_ant = LDA()
iso_ant_correls_reps = []
iso_ant_pval_reps = []
for i in np.arange(len(res_ind)):
    lda_ant.fit(emi_ho_reps_inverse[i], emi_ho_binding_inverse[i].iloc[:,0])
    iso_ant_transform_ho = pd.DataFrame(lda_ant.transform(iso_ho_reps[i]))
    iso_ant_correls_reps.append(abs(sc.stats.spearmanr(iso_ant_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,1])[0]))
    iso_ant_pval_reps.append(abs(sc.stats.spearmanr(iso_ant_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,1])[1]))

lda_psy = LDA()
iso_psy_correls_reps = []
iso_psy_pval_reps = []
for i in np.arange(len(res_ind)):
    lda_psy.fit(emi_ho_reps_inverse[i], emi_ho_binding_inverse[i].iloc[:,1])
    iso_psy_transform_ho = pd.DataFrame(lda_psy.transform(iso_ho_reps[i]))
    iso_psy_correls_reps.append(abs(sc.stats.spearmanr(iso_psy_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,2])[0]))
    iso_psy_pval_reps.append(abs(sc.stats.spearmanr(iso_psy_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,2])[1]))

lda_ant = LDA()
iso_ant_correls_physchemvh = []
iso_ant_pval_physchemvh = []
for i in np.arange(len(res_ind)):
    lda_ant.fit(emi_ho_physchemvh_inverse[i], emi_ho_binding_inverse[i].iloc[:,0])
    iso_ant_transform_ho = pd.DataFrame(lda_ant.transform(iso_ho_physchemvh[i]))
    iso_ant_correls_physchemvh.append(abs(sc.stats.spearmanr(iso_ant_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,1])[0]))
    iso_ant_pval_physchemvh.append(abs(sc.stats.spearmanr(iso_ant_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,1])[1]))

lda_psy = LDA()
iso_psy_correls_physchemvh = []
iso_psy_pval_physchemvh = []
for i in np.arange(len(res_ind)):
    lda_psy.fit(emi_ho_physchemvh_inverse[i], emi_ho_binding_inverse[i].iloc[:,1])
    iso_psy_transform_ho = pd.DataFrame(lda_psy.transform(iso_ho_physchemvh[i]))
    iso_psy_correls_physchemvh.append(abs(sc.stats.spearmanr(iso_psy_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,2])[0]))
    iso_psy_pval_physchemvh.append(abs(sc.stats.spearmanr(iso_psy_transform_ho.iloc[:,0], iso_ho_binding[i].iloc[:,2])[1]))



