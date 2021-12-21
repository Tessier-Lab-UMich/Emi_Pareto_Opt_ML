# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:33:10 2021

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

cmap = plt.cm.get_cmap('bwr')
colormap9= np.array([cmap(0.15),cmap(0.85)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap9r= np.array([cmap(0.85),cmap(0.15)])
cmap9r = LinearSegmentedColormap.from_list("mycmap", colormap9r)


def split(word): 
    return [char for char in word]

res_ind = [32, 49, 54, 55, 56, 98, 100, 103]
res_aa = ['Y','R','R','R','G','A','W','Y']

def ho_seq_ind(seqs):
    holdout_seqs = []
    for i in np.arange(len(res_ind)):
        holdout = []
        for j in seqs:
            chars = list(j)
            if chars[res_ind[i]] == res_aa[i]:
                holdout.append(''.join(str(ii) for ii in chars))
        holdout_seqs.append(pd.DataFrame(holdout))
    return holdout_seqs

def ho_seq_ind_inverse(seqs):
    holdout_seqs = []
    for i in np.arange(len(res_ind)):
        holdout = []
        for j in seqs:
            chars = list(j)
            if chars[res_ind[i]] != res_aa[i]:
                holdout.append(''.join(str(ii) for ii in chars))
        holdout_seqs.append(pd.DataFrame(holdout))
    return holdout_seqs

def ho_binding(seq_ind, binding):
    holdout_binding = []
    for i in np.arange(len(res_ind)):
        holdout = binding.filter(items = seq_ind[i].iloc[:,0], axis = 0)
        holdout_binding.append(holdout)
    return holdout_binding


def ho_reps(seq_ind, reps):
    holdout_reps = []
    for i in np.arange(len(res_ind)):
        holdout = reps.filter(items = seq_ind[i].iloc[:,0], axis = 0)
        holdout_reps.append(holdout)
    return holdout_reps
        

alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)
one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

def ho_ohe(seq_ind):
    ohe = []
    for i in np.arange(len(res_ind)):
        enc = []
        holdout_ohe = []
        for i in seq_ind[i].iloc[:,0]:
            chars = le.transform(list(i))
            enc.append(chars)
        enc = pd.DataFrame(enc)
        for index, row in enc.iterrows():
            enc_row = np.array(row)
            let = enc_row.reshape(115,1)
            ohe_let = pd.DataFrame(one.transform(let))
            holdout_ohe.append(ohe_let.values.flatten())
        holdout_ohe = pd.DataFrame(np.stack(holdout_ohe))
        ohe.append(holdout_ohe)
    return ohe


residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)
def ho_physchemmut(seq_ind):
    holdout_biophys = []
    for k in np.arange(len(res_ind)):
        mutations = []
        for ii in seq_ind[k].iloc[:,0]:
            characters = list(ii)
            mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
        mutations = pd.DataFrame(mutations)
        mutations_biophys = []
        for i in mutations.iterrows():
            seq_mutations_biophys = []
            seq_mutations_biophys_stack = []
            for j in i[1]:
                seq_mutations_biophys.append(residue_info.loc[j,:].values)
            seq_mutations_biophys_stack = np.hstack(seq_mutations_biophys)
            mutations_biophys.append(seq_mutations_biophys_stack)
        mutations_biophys = pd.DataFrame(mutations_biophys)
        mutations_biophys.columns = ['33PosCharge','33NegCharge','33HM','33pI','33Atoms','33HBondAD','50PosCharge','50NegCharge','50HM','50pI','50Atoms','50HBondAD','55PosCharge','55NegCharge','55HM','55pI','55Atoms','55HBondAD','56PosCharge','56NegCharge','56HM','56pI','56Atoms','56HBondAD','57PosCharge','57NegCharge','57HM','57pI','57Atoms','57HBondAD','99PosCharge','99NegCharge','99HM','99pI','99Atoms','99HBondAD','101PosCharge','101NegCharge','101HM','101pI','101Atoms','101HBondAD','104PosCharge','104NegCharge','104HM','104pI','104Atoms','104HBondAD']
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR2 PosCharge'] = ((mutations_biophys.iloc[j,6]) + (mutations_biophys.iloc[j,12]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,24]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR2 NegCharge'] = ((mutations_biophys.iloc[j,7]) + (mutations_biophys.iloc[j,13]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,25]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR2 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,8]) + (mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,26]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR2 pI'] = ((mutations_biophys.iloc[j,9]) + (mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,19]) + (mutations_biophys.iloc[j,27]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR2 # Atoms'] = ((mutations_biophys.iloc[j,10]) + (mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,28]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR2 HBondA'] = ((mutations_biophys.iloc[j,11]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,21]) + (mutations_biophys.iloc[j,29]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR3 PosCharge'] = ((mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,42]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR3 NegCharge'] = ((mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,37]) + (mutations_biophys.iloc[j,43]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR3 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,44]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCR3 pI'] = ((mutations_biophys.iloc[j,33]) + (mutations_biophys.iloc[j,39]) + (mutations_biophys.iloc[j,45]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR3 # Atoms'] = ((mutations_biophys.iloc[j,34]) + (mutations_biophys.iloc[j,40]) + (mutations_biophys.iloc[j,46]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HCDR3 HBondA'] = ((mutations_biophys.iloc[j,35]) + (mutations_biophys.iloc[j,41]) + (mutations_biophys.iloc[j,47]))
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j,'PosCharge Score'] = (mutations_biophys.iloc[j,0] + mutations_biophys.iloc[j,6] + mutations_biophys.iloc[j,12] + mutations_biophys.iloc[j,18] + mutations_biophys.iloc[j,24] + mutations_biophys.iloc[j,30] + mutations_biophys.iloc[j,36] + mutations_biophys.iloc[j,42])
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j,'NegCharge'] = (mutations_biophys.iloc[j,1] + mutations_biophys.iloc[j,7] + mutations_biophys.iloc[j,13] + mutations_biophys.iloc[j,19] + mutations_biophys.iloc[j,25] + mutations_biophys.iloc[j,31] + mutations_biophys.iloc[j,37] + mutations_biophys.iloc[j,43])
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j,'Hydrophobic Moment'] = (mutations_biophys.iloc[j,2] + mutations_biophys.iloc[j,8] + mutations_biophys.iloc[j,14] + mutations_biophys.iloc[j,20] + mutations_biophys.iloc[j,26] + mutations_biophys.iloc[j,32] + mutations_biophys.iloc[j,38] + mutations_biophys.iloc[j,44])
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j,'pI'] = (mutations_biophys.iloc[j,3] + mutations_biophys.iloc[j,9] + mutations_biophys.iloc[j,15] + mutations_biophys.iloc[j,21] + mutations_biophys.iloc[j,27] + mutations_biophys.iloc[j,33] + mutations_biophys.iloc[j,39] + mutations_biophys.iloc[j,45])
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, '# Atoms'] = ((mutations_biophys.iloc[j,4]) + (mutations_biophys.iloc[j,10]) + (mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,22]) + mutations_biophys.iloc[j,28] + mutations_biophys.iloc[j,34] + mutations_biophys.iloc[j,40] + mutations_biophys.iloc[j,46])
        for i in mutations_biophys.iterrows():
            j = i[0]
            mutations_biophys.loc[j, 'HBondA'] = ((mutations_biophys.iloc[j,5]) + (mutations_biophys.iloc[j,11]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,23]) + mutations_biophys.iloc[j,29] + mutations_biophys.iloc[j,35] + mutations_biophys.iloc[j,41] + mutations_biophys.iloc[j,47])
        holdout_biophys.append(mutations_biophys)
    return holdout_biophys

def ho_physchemvh(seq_ind, seq_binding):
    holdout_physchemvh = []
    for k in np.arange(len(res_ind)):
        res_counts = pd.DataFrame(index = alph_letters)
        for i in seq_ind[k].iloc[:,0]:
            characters = pd.Series(list(i))
            res_counts = pd.concat([res_counts, characters.value_counts()], axis = 1, ignore_index = False)
        res_counts.fillna(0, inplace = True)
        res_counts = res_counts.T
        hydrophobicity = []    
        for column in res_counts:
            hydros = []
            for index, row in res_counts.iterrows():
                hydros.append(row[column]*residue_info.loc[column, 'Hydropathy Score'])
            hydrophobicity.append(hydros)
        hydrophobicity = pd.DataFrame(hydrophobicity).T
        hydrophobicity['ave'] = hydrophobicity.sum(axis = 1)/115
        res_counts['Hydro'] = res_counts['A'] +  res_counts['I'] +  res_counts['L']+  res_counts['F']+  res_counts['V']
        res_counts['Amph'] = res_counts['W'] +  res_counts['Y']+  res_counts['M']
        res_counts['Polar'] = res_counts['Q'] +  res_counts['N'] + res_counts['S'] +  res_counts['T'] +  res_counts['C']+  res_counts['M']
        res_counts['Charged'] =  res_counts['R'] +  res_counts['K'] + res_counts['D'] +  res_counts['E'] +  res_counts['H']
        res_counts.reset_index(drop = True, inplace = True)
        physchemvh = pd.concat([res_counts, seq_binding['pI_seq'].reset_index(drop = True, inplace = True), hydrophobicity['ave']], axis = 1, ignore_index = False)
        holdout_physchemvh.append(physchemvh)
    return holdout_physchemvh

