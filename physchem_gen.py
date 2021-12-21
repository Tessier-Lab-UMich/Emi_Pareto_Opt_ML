# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:19:35 2021

@author: makow
"""

import numpy as np
import pandas as pd

residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)
res_ind = [32, 49, 54, 55, 56, 98, 100, 103]
res_aa = ['Y','R','R','R','G','A','W','Y']
alph = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))

def physchemvh_gen(seqs_binding, pI):
    res_counts = pd.DataFrame(index = alph)
    for i in seqs_binding.index:
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
    physchemvh = pd.concat([res_counts, seqs_binding['pI_seq'].reset_index(drop = True, inplace = True), hydrophobicity['ave'], pI], axis = 1, ignore_index = False)
    return physchemvh


