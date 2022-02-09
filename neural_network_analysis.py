# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:43:44 2022

@author: pkinn
"""

import tensorflow as tf
from holdout_utils import *
from physchem_gen import physchemvh_gen
from onehot_gen import onehot_gen, shiftedColorMap
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


#%% Define NN model
class projectorDecider(Model):
    def __init__(self, projDim, inputDim):
        super(projectorDecider, self).__init__()
        self.projDim = projDim
        self.inputDim = inputDim
        self.projector = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.inputDim)),
            tf.keras.layers.Dense(self.projDim)])
        self.decider = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.projDim,)),
            tf.keras.layers.Dense(2)])
    def call(self, x):
        projected = self.projector(x)
        decided = self.decider(projected)
        return decided
    
    
#%% Load data into dataframes
emi_binding = pd.read_csv("emi_binding.csv", header = 0, index_col = 0)
iso_binding = pd.read_csv("iso_binding.csv", header = 0, index_col = 0)
igg_binding = pd.read_csv("igg_binding.csv", header = 0, index_col = 0)

#%% generate OHE features
emi_onehot = onehot_gen(emi_binding)
iso_onehot = onehot_gen(iso_binding)
igg_onehot = onehot_gen(igg_binding)

#%% Import unirep features
emi_reps = pd.read_csv("emi_reps.csv", header = 0, index_col = 0)
iso_reps = pd.read_csv("iso_reps.csv", header = 0, index_col = 0)
igg_reps = pd.read_csv("igg_reps.csv", header = 0, index_col = 0)

#%% Generate Physicochemical features
emi_pI = pd.read_csv("emi_pI.txt", sep = '\t', header = None, index_col = None)
iso_pI = pd.read_csv("iso_pI.txt", sep = '\t',header = None, index_col = None)
igg_pI = pd.read_csv("igg_pI.txt", sep = '\t',header = None, index_col = None)

emi_physvh = physchemvh_gen(emi_binding, emi_pI.iloc[:,1])
iso_physvh = physchemvh_gen(iso_binding, iso_pI.iloc[:,1])
igg_physvh = physchemvh_gen(igg_binding, igg_pI.iloc[:,1])

#%% Train "final" models
targetsAnt = emi_binding['ANT Binding'].values
targetsPsy = emi_binding['OVA Binding'].values
batch_sz = 50
nepochs = 20

pdAntOH = projectorDecider(1, emi_onehot.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdAntOH.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntOH.fit(emi_onehot.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)

pdPsyOH = projectorDecider(1, emi_onehot.shape[1])
pdPsyOH.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyOH.fit(emi_onehot.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)

pdAntUR = projectorDecider(1, emi_reps.shape[1])
pdAntUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntUR.fit(emi_reps.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)

pdPsyUR = projectorDecider(1, emi_reps.shape[1])
pdPsyUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyUR.fit(emi_reps.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)

pdAntPC = projectorDecider(1, emi_physvh.shape[1])
pdAntPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntPC.fit(emi_physvh.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)

pdPsyPC = projectorDecider(1, emi_physvh.shape[1])
pdPsyPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyPC.fit(emi_physvh.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)

#%% Project isolated OHE features and compare w/ data
plt.close('all')
projIsoAntOH = pdAntOH.projector(iso_onehot.values)
plt.figure()
plt.scatter(projIsoAntOH, iso_binding['ANT Binding'].values)
plt.xlabel('NN Projection')
plt.ylabel('Measured ANT binding')
antCorr = sc.stats.spearmanr(projIsoAntOH, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
target = 'Antigen'
ftSet = 'Onehot'
titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
plt.title(titleText)


projIsoPsyOH = pdPsyOH.projector(iso_onehot.values)
plt.figure()
plt.scatter(projIsoPsyOH, iso_binding['OVA Binding'].values)
plt.xlabel('NN Projection')
plt.ylabel('Measured OVA binding')
ovaCorr = sc.stats.spearmanr(projIsoPsyOH, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
target = 'Ova'
ftSet = 'Onehot'
titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
plt.title(titleText)


#%%
projIsoAntUR = pdAntUR.projector(iso_reps.values)
plt.figure()
plt.scatter(projIsoAntUR, iso_binding['ANT Binding'].values)
antCorr = sc.stats.spearmanr(projIsoAntUR, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
plt.xticks([-0.35, -0.30, -0.25], [-0.35, -0.30, -0.25], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

projIsoPsyUR = pdPsyUR.projector(iso_reps.values)
plt.figure()
plt.scatter(projIsoPsyUR, iso_binding['OVA Binding'].values)
ovaCorr = sc.stats.spearmanr(projIsoPsyUR, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
plt.xticks([-0.08, -0.04, 0.0, 0.04], [-0.08, -0.04, 0.0, 0.04], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


projIsoAntPC = pdAntPC.projector(iso_physvh.values)
plt.figure()
plt.scatter(projIsoAntPC, iso_binding['ANT Binding'].values)
antCorr = sc.stats.spearmanr(projIsoAntPC, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
plt.xticks([-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

projIsoPsyPC = pdPsyPC.projector(iso_physvh.values)
plt.figure()
plt.scatter(projIsoPsyPC, iso_binding['OVA Binding'].values)
ovaCorr = sc.stats.spearmanr(projIsoPsyPC, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


#%%
projIsoAntUR = pdAntUR.projector(igg_reps.values)
plt.figure()
plt.scatter(projIsoAntUR, iso_binding['ANT Binding'].values)
antCorr = sc.stats.spearmanr(projIsoAntUR, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
plt.xticks([-0.35, -0.30, -0.25], [-0.35, -0.30, -0.25], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

projIsoPsyUR = pdPsyUR.projector(igg_reps.values)
plt.figure()
plt.scatter(projIsoPsyUR, iso_binding['OVA Binding'].values)
ovaCorr = sc.stats.spearmanr(projIsoPsyUR, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
plt.xticks([-0.08, -0.04, 0.0, 0.04], [-0.08, -0.04, 0.0, 0.04], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


projIsoAntPC = pdAntPC.projector(igg_physvh.values)
plt.figure()
plt.scatter(projIsoAntPC, iso_binding['ANT Binding'].values)
antCorr = sc.stats.spearmanr(projIsoAntPC, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
plt.xticks([-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

projIsoPsyPC = pdPsyPC.projector(iso_physvh.values)
plt.figure()
plt.scatter(projIsoPsyPC, iso_binding['OVA Binding'].values)
ovaCorr = sc.stats.spearmanr(projIsoPsyPC, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

