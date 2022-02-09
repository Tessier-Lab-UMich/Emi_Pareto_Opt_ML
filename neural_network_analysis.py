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

class deepProjectorDecider(Model):
    def __init__(self, projDim, inputDim, intermedDim):
        super(deepProjectorDecider, self).__init__()
        self.projDim = projDim
        self.inputDim = inputDim
        self.intermedDim = intermedDim
        self.projector = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.inputDim)),
            tf.keras.layers.Dense(self.intermedDim),
            tf.keras.layers.Dense(self.projDim)])
        self.decider = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (self.projDim,)),
            tf.keras.layers.Dense(self.intermedDim),
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
from cvTrain import cvTrain
targetsAnt = emi_binding['ANT Binding'].values
targetsPsy = emi_binding['OVA Binding'].values
batch_sz = 50
nepochs = 50
nepochsUR = 500
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
                        epochs = nepochsUR,
                        verbose = 1)

pdPsyUR = projectorDecider(1, emi_reps.shape[1])
pdPsyUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyUR.fit(emi_reps.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochsUR,
                        verbose = 1)


# pdAntURDeep= deepProjectorDecider(1, emi_reps.shape[1], 20)
# pdAntURDeep.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
# pdAntURDeep.fit(emi_reps.values, targetsAnt, 
#                         batch_size = batch_sz, 
#                         epochs = 1000,
#                         verbose = 1)

# pdPsyURDeep = deepProjectorDecider(1, emi_reps.shape[1], 20)
# pdPsyURDeep.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
# pdPsyURDeep.fit(emi_reps.values, targetsPsy, 
#                         batch_size = batch_sz, 
#                         epochs = 1000,
#                         verbose = 1)


pdAntPC = projectorDecider(1, emi_physvh.shape[1])
pdAntPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntPC.fit(emi_physvh.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = 1000,
                        verbose = 1)

pdPsyPC = projectorDecider(1, emi_physvh.shape[1])
pdPsyPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyPC.fit(emi_physvh.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1000)

#%% What is this doing?
projEmiAntOH = pd.DataFrame(pdAntOH.projector(emi_onehot.values)).set_index(emi_binding.index)
projEmiPsyOH = pd.DataFrame(pdPsyOH.projector(emi_onehot.values)).set_index(emi_binding.index)
projEmiAntUR = pd.DataFrame(pdAntUR.projector(emi_reps.values)).set_index(emi_binding.index)
projEmiPsyUR = pd.DataFrame(pdPsyUR.projector(emi_reps.values)).set_index(emi_binding.index)
projEmiAntPC = pd.DataFrame(pdAntPC.projector(emi_physvh.values)).set_index(emi_binding.index)
projEmiPsyPC = pd.DataFrame(pdPsyPC.projector(emi_physvh.values)).set_index(emi_binding.index)

# projEmiAntURDeep = pd.DataFrame(pdAntURDeep.projector(emi_reps.values)).set_index(emi_binding.index)
# projEmiPsyURDeep = pd.DataFrame(pdPsyURDeep.projector(emi_reps.values)).set_index(emi_binding.index)

#%% Visualize classification accuracy and projection
plt.close('all')
plt.figure()
plt.subplot(3,2,1)
sns.distplot(-1*projEmiAntOH.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(-1*projEmiAntOH.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
accFinal = pdAntOH.history.history['accuracy'][-1]
target = 'Antigen'
ftSet = 'Onehot'
titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
plt.title(titleText)
# plt.xticks([-30, -20, -10, 0, 10], [-30, -20, -10, 0, 10], fontsize = 26)
# plt.yticks([0.0, 0.04, 0.08, 0.12, 0.16], [0.0, 0.04, 0.08, 0.12, 0.16], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

# plt.figure()
plt.subplot(3,2,2)

sns.distplot(-1*projEmiPsyOH.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(-1*projEmiPsyOH.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
accFinal = pdPsyOH.history.history['accuracy'][-1]
target = 'Ova'
ftSet = 'Onehot'
titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
plt.title(titleText)
# plt.xticks([-5, 0, 5], [-5, 0, 5], fontsize = 26)
# plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

# plt.figure()
plt.subplot(3,2,3)

sns.distplot(-1*projEmiAntUR.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(-1*projEmiAntUR.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
accFinal = pdAntUR.history.history['accuracy'][-1]
target = 'Antigen'
ftSet = 'UniRep'
titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
plt.title(titleText)
# plt.xticks([-30, -20, -10, 0, 10], [-30, -20, -10, 0, 10], fontsize = 26)
# plt.yticks([0.0, 0.04, 0.08, 0.12, 0.16], [0.0, 0.04, 0.08, 0.12, 0.16], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

# plt.figure()
plt.subplot(3,2,4)

sns.distplot(-1*projEmiPsyUR.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(-1*projEmiPsyUR.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
accFinal = pdPsyUR.history.history['accuracy'][-1]
target = 'Ova'
ftSet = 'UniRep'
titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
plt.title(titleText)
# plt.xticks([-5, 0, 5], [-5, 0, 5], fontsize = 26)
# plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

# plt.figure()
plt.subplot(3,2,5)
sns.distplot(-1*projEmiAntPC.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(-1*projEmiAntPC.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
accFinal = pdAntPC.history.history['accuracy'][-1]
target = 'Antigen'
ftSet = 'PhysChem'
titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
plt.title(titleText)

# plt.xticks([-30, -20, -10, 0, 10], [-30, -20, -10, 0, 10], fontsize = 26)
# plt.yticks([0.0, 0.04, 0.08, 0.12, 0.16], [0.0, 0.04, 0.08, 0.12, 0.16], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

# plt.figure()
plt.subplot(3,2,6)
sns.distplot(-1*projEmiPsyPC.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(-1*projEmiPsyPC.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
accFinal = pdPsyPC.history.history['accuracy'][-1]
target = 'Ova'
ftSet = 'PhysChem'
titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
plt.title(titleText)

# plt.xticks([-5, 0, 5], [-5, 0, 5], fontsize = 26)
# plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

# plt.figure()
# plt.subplot(4,2,7)
# sns.distplot(-1*projEmiAntURDeep.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
# sns.distplot(-1*projEmiAntURDeep.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
# accFinal = pdAntURDeep.history.history['accuracy'][-1]
# target = 'Antigen'
# ftSet = 'Deep UR'
# titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
# plt.title(titleText)

# # plt.xticks([-30, -20, -10, 0, 10], [-30, -20, -10, 0, 10], fontsize = 26)
# # plt.yticks([0.0, 0.04, 0.08, 0.12, 0.16], [0.0, 0.04, 0.08, 0.12, 0.16], fontsize = 26)
# plt.ylabel('')
# plt.xlabel('')

# # plt.figure()
# plt.subplot(4,2,8)
# sns.distplot(-1*projEmiPsyURDeep.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
# sns.distplot(-1*projEmiPsyURDeep.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
# accFinal = pdAntURDeep.history.history['accuracy'][-1]
# target = 'Ova'
# ftSet = 'Deep UR'
# titleText = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
# plt.title(titleText)

# # plt.xticks([-5, 0, 5], [-5, 0, 5], fontsize = 26)
# # plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 26)
# plt.ylabel('')
# plt.xlabel('')

#%% Project isolated OHE features and compare w/ data
# plt.close('all')
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

projIsoAntUR = pdAntUR.projector(iso_reps.values)
plt.figure()
plt.scatter(projIsoAntUR, iso_binding['ANT Binding'].values)
plt.xlabel('NN Projection')
plt.ylabel('Measured ANT binding')
antCorr = sc.stats.spearmanr(projIsoAntUR, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
target = 'Antigen'
ftSet = 'UniRep'
titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
plt.title(titleText)


projIsoPsyUR = pdPsyUR.projector(iso_reps.values)
plt.figure()
plt.scatter(projIsoPsyUR, iso_binding['OVA Binding'].values)
plt.xlabel('NN Projection')
plt.ylabel('Measured OVA binding')
ovaCorr = sc.stats.spearmanr(projIsoPsyUR, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
target = 'Ova'
ftSet = 'UniRep'
titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
plt.title(titleText)


projIsoAntPC = pdAntPC.projector(iso_physvh.values)
plt.figure()
plt.scatter(projIsoAntPC, iso_binding['ANT Binding'].values)
plt.xlabel('NN Projection')
plt.ylabel('Measured ANT binding')
antCorr = sc.stats.spearmanr(projIsoAntPC, iso_binding['ANT Binding'].values)
corrRho = np.round(antCorr.correlation, 2)
corrP = np.ceil(np.log10(antCorr.pvalue))
target = 'Antigen'
ftSet = 'PhysVH'
titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
plt.title(titleText)


projIsoPsyPC = pdPsyPC.projector(iso_physvh.values)
plt.figure()
plt.scatter(projIsoPsyPC, iso_binding['OVA Binding'].values)
plt.xlabel('NN Projection')
plt.ylabel('Measured OVA binding')
ovaCorr = sc.stats.spearmanr(projIsoPsyPC, iso_binding['OVA Binding'].values)
corrRho = np.round(ovaCorr.correlation, 2)
corrP = np.ceil(np.log10(ovaCorr.pvalue))
target = 'Ova'
ftSet = 'PhysVH'
titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
plt.title(titleText)

# projIsoAntURDeep = pdAntURDeep.projector(iso_reps.values)
# plt.figure()
# plt.scatter(projIsoAntURDeep, iso_binding['ANT Binding'].values)
# plt.xlabel('NN Projection')
# plt.ylabel('Measured ANT binding')
# antCorr = sc.stats.spearmanr(projIsoAntUR, iso_binding['ANT Binding'].values)
# corrRho = np.round(antCorr.correlation, 2)
# corrP = np.ceil(np.log10(antCorr.pvalue))
# target = 'Antigen'
# ftSet = 'UniRep Deep'
# titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
# plt.title(titleText)


# projIsoPsyURDeep = pdPsyURDeep.projector(iso_reps.values)
# plt.figure()
# plt.scatter(projIsoPsyURDeep, iso_binding['OVA Binding'].values)
# plt.xlabel('NN Projection')
# plt.ylabel('Measured OVA binding')
# ovaCorr = sc.stats.spearmanr(projIsoPsyUR, iso_binding['OVA Binding'].values)
# corrRho = np.round(ovaCorr.correlation, 2)
# corrP = np.ceil(np.log10(ovaCorr.pvalue))
# target = 'Ova'
# ftSet = 'UniRep Deep'
# titleText = f'{target}/{ftSet} rho = {corrRho} log10(p) < {corrP}'
# plt.title(titleText)