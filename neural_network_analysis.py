# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:43:44 2022

@author: pkinn
"""
"""
This script runs all neural network code to generate figures S5 and S9. 

Variables are identified based on the dataset they represent (isolated clones 
= Iso, IGG, or emi = sorted clones), the feature set used (OH = one-hot, PC = 
physchem, UR = unirep), and the binding target (Ant = antigen, 
Ova/Psy = specificity reagent). 

Thus, projEmiAntOH is the neural network projection of the 4000 sorted
sequences based on a NN trained to predict Antigen binding with One-hot
features. 

"""


import tensorflow as tf
from holdout_utils import *
from physchem_gen import physchemvh_gen
from onehot_gen import onehot_gen, shiftedColorMap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import os



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

#%% Make NN Instances and load weights
nIntermedNodes = 20 #This can't be changed unless different weight files are used
wtPath = f'deepNNWeights{os.sep}node{nIntermedNodes}{os.sep}'

pdAntOH = deepProjectorDecider(1, emi_onehot.shape[1], nIntermedNodes)
wtName = 'pdAntOH_wts'
pdAntOH.load_weights(wtPath+wtName)

pdPsyOH = deepProjectorDecider(1, emi_onehot.shape[1], nIntermedNodes)
wtName = 'pdPsyOH_wts'
pdPsyOH.load_weights(wtPath+wtName)

pdAntUR = deepProjectorDecider(1, emi_reps.shape[1], nIntermedNodes)
wtName = 'pdAntUR_wts'
pdAntUR.load_weights(wtPath+wtName)

pdPsyUR = deepProjectorDecider(1, emi_reps.shape[1], nIntermedNodes)
wtName = 'pdPsyUR_wts'
pdPsyUR.load_weights(wtPath+wtName)

pdAntPC = deepProjectorDecider(1, emi_physvh.shape[1], nIntermedNodes)
wtName = 'pdAntPC_wts'
pdAntPC.load_weights(wtPath+wtName)

pdPsyPC = deepProjectorDecider(1, emi_physvh.shape[1], nIntermedNodes)
wtName = 'pdPsyPC_wts'
pdPsyPC.load_weights(wtPath+wtName)


#%% 
#this is making the indices agree so seaborn will plot them
projEmiAntOH = pd.DataFrame(pdAntOH.projector(emi_onehot.values)).set_index(emi_binding.index)
projEmiPsyOH = pd.DataFrame(pdPsyOH.projector(emi_onehot.values)).set_index(emi_binding.index)
projEmiAntUR = pd.DataFrame(pdAntUR.projector(emi_reps.values)).set_index(emi_binding.index)
projEmiPsyUR = pd.DataFrame(pdPsyUR.projector(emi_reps.values)).set_index(emi_binding.index)
projEmiAntPC = pd.DataFrame(pdAntPC.projector(emi_physvh.values)).set_index(emi_binding.index)
projEmiPsyPC = pd.DataFrame(pdPsyPC.projector(emi_physvh.values)).set_index(emi_binding.index)

#%% Visualize classification accuracy and projection
#Updated 
plt.figure()
sns.distplot(-1*projEmiAntOH.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(-1*projEmiAntOH.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
predValsEmiAntOH = np.argmax(pdAntOH.decider(pdAntOH.projector(emi_onehot.values)), 1)
#accFinalEmiAntOH = sum(predValsEmiAntOH==targetsAnt)/len(predValsEmiAntOH)
target = 'Antigen'
ftSet = 'Onehot'
#titleTextEmiAntOH = f'{target}/{ftSet} Final Accuracy: {np.round(accFinalEmiAntOH,3)}'
#plt.title(titleTextEmiAntOH)
EmiAntOHX = [-10, -5, 0, 5]
plt.xticks(EmiAntOHX, EmiAntOHX, fontsize = 26)
plt.xlim(-13, 6)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 26)
plt.ylabel('')
plt.xlabel('')


plt.figure()
sns.distplot(projEmiPsyOH.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(projEmiPsyOH.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
predValsEmiPsyOH = np.argmax(pdPsyOH.decider(pdPsyOH.projector(emi_onehot.values)), 1)
#accFinal = sum(predValsEmiPsyOH==targetsPsy)/len(predValsEmiPsyOH)
target = 'Ova'
ftSet = 'Onehot'
#titleTextEmiPsyOH = f'{target}/{ftSet} Final Accuracy: {np.round(accFinal,3)}'
#plt.title(titleTextEmiPsyOH)
EmiPsyOHX = [-20, -10, 0, 10,  20]
plt.xticks(EmiPsyOHX, EmiPsyOHX, fontsize = 26)
plt.yticks([0.00, 0.05,  0.10, 0.15], [0.00, 0.05,  0.10, 0.15], fontsize = 26)
plt.ylabel('')
plt.xlabel('')


plt.figure()
sns.distplot(projEmiAntUR.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(projEmiAntUR.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
predValsEmiAntUR = np.argmax(pdAntUR.decider(pdAntUR.projector(emi_reps.values)), 1)
#accFinalEmiAntUR = sum(predValsEmiAntUR==targetsAnt)/len(predValsEmiAntUR)
target = 'Antigen'
ftSet = 'UniRep'
#titleTextEmiAntUR = f'{target}/{ftSet} Final Accuracy: {np.round(accFinalEmiAntUR,3)}'
EmiAntURX = [-1.5,-1., -0.5, 0.0, 0.5, 1.0]
#plt.title(titleTextEmiAntUR)
plt.xticks(EmiAntURX,EmiAntURX, fontsize = 26)
plt.xlim(-1.9, 1.0)
plt.yticks([0.0, 0.5, 1.0, 1.5,2.0], [0.0, 0.5, 1.0, 1.5,2.0], fontsize = 26)
plt.ylabel('')
plt.xlabel('')


plt.figure()
sns.distplot(projEmiPsyUR.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(projEmiPsyUR.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
predValsEmiPsyUR = np.argmax(pdPsyUR.decider(pdPsyUR.projector(emi_reps.values)), 1)
#accFinalEmiPsyUR = sum(predValsEmiPsyUR==targetsPsy)/len(predValsEmiPsyUR)
target = 'Ova'
ftSet = 'UniRep'
#titleTextEmiPsyUR = f'{target}/{ftSet} Final Accuracy: {np.round(accFinalEmiPsyUR,3)}'
#plt.title(titleTextEmiPsyUR)
EmiPsyURX = [-2, -1, 0, 1, 2]
plt.xticks(EmiPsyURX, EmiPsyURX, fontsize = 26)
plt.yticks([0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0, 1.5], fontsize = 26)
plt.ylabel('')
plt.xlabel('')

plt.figure()
sns.distplot(-1*projEmiAntPC.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(-1*projEmiAntPC.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
predValsEmiAntPC = np.argmax(pdAntPC.decider(pdAntPC.projector(emi_physvh.values)), 1)
#accFinalEmiAntPC = sum(predValsEmiAntPC==targetsAnt)/len(predValsEmiAntPC)
target = 'Antigen'
ftSet = 'PhysChem'
#titleTextEmiAntPC = f'{target}/{ftSet} Final Accuracy: {np.round(accFinalEmiAntPC,3)}'
#plt.title(titleTextEmiAntPC)
EmiAntPCX = [-8, -4, 0, 4]
plt.xticks(EmiAntPCX, EmiAntPCX, fontsize = 26)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 26)
plt.ylabel('')
plt.xlabel('')


plt.figure()
sns.distplot(-1*projEmiPsyPC.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(-1*projEmiPsyPC.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
predValsEmiPsyPC = np.argmax(pdPsyPC.decider(pdPsyPC.projector(emi_physvh.values)), 1)
#accFinalEmiPsyPC = sum(predValsEmiPsyPC==targetsPsy)/len(predValsEmiPsyPC)
target = 'Ova'
ftSet = 'PhysChem'
#titleTextEmiPsyPC = f'{target}/{ftSet} Final Accuracy: {np.round(accFinalEmiPsyPC,3)}'
#plt.title(titleTextEmiPsyPC)
EmiPsyPCX = [-20, -10, 0, 10,  20]
plt.xticks(EmiPsyPCX, EmiPsyPCX, fontsize = 26)
plt.yticks([0.00, 0.05, 0.10, 0.15], [0.00, 0.05, 0.10, 0.15], fontsize = 26)
plt.ylabel('')
plt.xlabel('')


#%% Project isolated OHE features and compare w/ data
#updated
projIsoAntOH = pdAntOH.projector(iso_onehot.values)
decIsoAntOH = np.argmax(pdAntOH.decider(np.array(projIsoAntOH)), 1)
plt.figure()
plt.scatter(-1*projIsoAntOH, iso_binding['ANT Binding'].values, c = decIsoAntOH, cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(-1*projIsoAntOH[125], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
#plt.xlabel('NN Projection')
#plt.ylabel('Measured ANT binding')
print('pearson correlation and p-value for Iso Ant PC')
print(sc.stats.pearsonr(np.array(projIsoAntOH).flatten(), iso_binding['ANT Binding'].values))
# plt.title('Iso Ant OH')
plt.xticks(fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


projIsoPsyOH = pdPsyOH.projector(iso_onehot.values)
decIsoPsyOH = np.argmax(pdPsyOH.decider(np.array(projIsoPsyOH)), 1)
plt.figure()
plt.scatter(projIsoPsyOH, iso_binding['OVA Binding'].values, c = decIsoPsyOH, cmap = cmap9, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(projIsoPsyOH[125], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
#plt.xlabel('NN Projection')
#plt.ylabel('Measured OVA binding')
print('pearson correlation and p-value for Iso Psy OH')
print(sc.stats.pearsonr(np.array(projIsoPsyOH).flatten(), iso_binding['OVA Binding'].values))
#plt.title('Iso Psy OH')
plt.xticks(fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


projIsoAntUR = pdAntUR.projector(iso_reps.values)
decIsoAntUR = np.argmax(pdAntUR.decider(np.array(projIsoAntUR)), 1)
plt.figure()
plt.scatter(projIsoAntUR, iso_binding['ANT Binding'].values, c = decIsoAntUR, cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(projIsoAntUR[125], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
#plt.xlabel('NN Projection')
#plt.ylabel('Measured ANT binding')
print('pearson correlation and p-value for Iso Ant UR')
print(sc.stats.pearsonr(np.array(projIsoAntUR).flatten(), iso_binding['ANT Binding'].values))
#plt.title('Iso Ant UR')
EmiAntURX = [-1.0, -0.5, 0.0, 0.5]
plt.xticks(EmiAntURX, EmiAntURX, fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


projIsoPsyUR = pdPsyUR.projector(iso_reps.values)
decIsoPsyUR = np.argmax(pdAntUR.decider(np.array(projIsoPsyUR)), 1)
plt.figure()
plt.scatter(projIsoPsyUR, iso_binding['OVA Binding'].values, c = decIsoPsyUR, cmap = cmap9, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(projIsoPsyUR[125], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
#plt.xlabel('NN Projection')
#plt.ylabel('Measured OVA binding')
print('pearson correlation and p-value for Iso Psy UR')
print(sc.stats.pearsonr(np.array(projIsoPsyUR).flatten(), iso_binding['OVA Binding'].values))
#plt.title('Iso Psy UR')
EmiPsyURX = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
plt.xticks(EmiPsyURX, EmiPsyURX, fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

#Iso correlations, PhysChem features
projIsoAntPC = pdAntPC.projector(iso_physvh.values)
decIsoAntPC = np.argmax(pdAntPC.decider(np.array(projIsoAntPC)), 1)
plt.figure()
plt.scatter(-1*projIsoAntPC, iso_binding['ANT Binding'].values, c = decIsoAntPC, cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(-1*projIsoAntPC[125], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
#plt.xlabel('NN Projection')
#plt.ylabel('Measured ANT binding')
print('pearson correlation and p-value for Iso Ant PC')
print(sc.stats.pearsonr(np.array(projIsoAntPC).flatten(), iso_binding['ANT Binding'].values))
plt.title('Iso Ant PC')
EmiAntPCX = [-6, -4, -2, 0, 2, 4]
plt.xticks(EmiAntPCX, EmiAntPCX, fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


projIsoPsyPC = pdPsyPC.projector(iso_physvh.values)
decIsoPsyPC = np.argmax(pdPsyPC.decider(np.array(projIsoPsyPC)),1)
plt.figure()
plt.scatter(-1*projIsoPsyPC, iso_binding['OVA Binding'].values, c = decIsoPsyPC, cmap = cmap9, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(-1*projIsoPsyPC[125], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
#plt.xlabel('NN Projection')
#plt.ylabel('Measured Ova Binding)
print('pearson correlation and p-value for Iso Psy PC')
print(sc.stats.pearsonr(np.array(projIsoPsyPC).flatten(), iso_binding['OVA Binding'].values))
# plt.title('Iso Psy PC')
plt.xticks(EmiPsyPCX, EmiPsyPCX, fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


#%% Project out of library sequences
projIggAntOH = pd.DataFrame(pdAntOH.projector(igg_onehot.values).numpy()).set_index(igg_binding.index)
projIggPsyOH = pd.DataFrame(pdPsyOH.projector(igg_onehot.values).numpy()).set_index(igg_binding.index)
projIggAntUR = pd.DataFrame(pdAntUR.projector(igg_reps.values).numpy()).set_index(igg_binding.index)
projIggPsyUR = pd.DataFrame(pdPsyUR.projector(igg_reps.values).numpy()).set_index(igg_binding.index)
projIggAntPC = pd.DataFrame(pdAntPC.projector(igg_physvh.values).numpy()).set_index(igg_binding.index)
projIggPsyPC = pd.DataFrame(pdPsyPC.projector(igg_physvh.values).numpy()).set_index(igg_binding.index)

#%%
#Pareto plots
#
plt.figure()
plt.scatter(-1*projEmiAntOH, projEmiPsyOH, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(-1*projIggAntOH.iloc[0:41,0], projIggPsyOH.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(-1*projIggAntOH.iloc[41:42,0], projIggPsyOH.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-10, -5, 0, 5], [-10, -5, 0, 5], fontsize = 26)
plt.yticks([-20, -10, 0, 10, 20], [-20, -10, 0, 10, 20], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(projEmiAntUR, projEmiPsyUR, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(projIggAntUR.iloc[0:41,0], projIggPsyUR.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(projIggAntUR.iloc[41:42,0], projIggPsyUR.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0], fontsize = 26)
plt.yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(-1*projEmiAntPC, -1*projEmiPsyPC, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(-1*projIggAntPC.iloc[0:41,0], -1*projIggPsyPC.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(-1*projIggAntPC.iloc[41:42,0], -1*projIggPsyPC.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4], [-6, -4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([-15, -5, 5, 15], [-15, -5, 5,  15], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(-1*projEmiAntOH, projEmiPsyOH, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(-1*projIggAntOH.loc[igg_binding['Blosum62'] == 1,0], projIggPsyOH.loc[igg_binding['Blosum62'] == 1,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(-1*projIggAntOH.iloc[41:42,0], projIggPsyOH.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.scatter(-1*projIggAntOH.iloc[8,0], projIggPsyOH.iloc[8,0], c = 'orange', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-10, -5, 0, 5], [-10, -5, 0, 5], fontsize = 26)
plt.yticks([-20, -10, 0, 10, 20], [-20, -10, 0, 10, 20], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(projEmiAntUR, projEmiPsyUR, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(projIggAntUR.loc[igg_binding['Blosum62'] == 1,0], projIggPsyUR.loc[igg_binding['Blosum62'] == 1,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(projIggAntUR.iloc[41:42,0], projIggPsyUR.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.scatter(projIggAntUR.iloc[8,0], projIggPsyUR.iloc[8,0], c = 'orange', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0], fontsize = 26)
plt.yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(-1*projEmiAntPC, -1*projEmiPsyPC, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(-1*projIggAntPC.loc[igg_binding['Blosum62'] == 1,0], -1*projIggPsyPC.loc[igg_binding['Blosum62'] == 1,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(-1*projIggAntPC.iloc[41:42,0], -1*projIggPsyPC.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.scatter(-1*projIggAntPC.iloc[8,0], -1*projIggPsyPC.iloc[8,0], c = 'orange', s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-6, -4, -2, 0, 2, 4], [-6, -4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([-15, -5, 5, 15], [-15, -5, 5,  15], fontsize = 26)
plt.ylabel('')


#%% Correlations with IGGs
plt.figure()
plt.errorbar(-1*projIggAntOH.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(-1*projIggAntOH.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(-1*projIggAntOH.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([ 1, 2, 3, 4], [1, 2, 3, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)
print('pearson correlation and p-value for Igg Ant OH')
print(sc.stats.pearsonr(projIggAntOH.iloc[0:41,0], igg_binding.iloc[0:41,1].values))

plt.figure()
plt.errorbar(projIggPsyOH.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(projIggPsyOH.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(projIggPsyOH.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([0.0, 2.5, 5.0, 7.5, 10.0], [0.0, 2.5, 5.0, 7.5, 10.0], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print('pearson correlation and p-value for Igg Psy OH')
print(sc.stats.pearsonr(projIggPsyOH.iloc[0:41,0], igg_binding.iloc[0:41,2].values))


plt.figure()
plt.errorbar(projIggAntUR.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(projIggAntUR.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(projIggAntUR.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-0.25, 0.00, 0.25,0.50, 0.75], [-0.25, 0.00, 0.25,0.50, 0.75], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)
print('pearson correlation and p-value for Igg Ant UR')
print(sc.stats.pearsonr(projIggAntUR.iloc[0:41,0], igg_binding.iloc[0:41,1].values))

plt.figure()
plt.errorbar(projIggPsyUR.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(projIggPsyUR.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(projIggPsyUR.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0, 1.5], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print('pearson correlation and p-value for Igg Psy UR')
print(sc.stats.pearsonr(projIggPsyUR.iloc[0:41,0], igg_binding.iloc[0:41,2].values))


plt.figure()
plt.errorbar(-1*projIggAntPC.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(-1*projIggAntPC.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(-1*projIggAntPC.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-1, 0, 1, 2, 3, 4], [-1, 0, 1, 2, 3, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)
print('pearson correlation and p-value for Igg Ant PC')
print(sc.stats.pearsonr(projIggAntPC.iloc[0:41,0], igg_binding.iloc[0:41,1].values))

plt.figure()
plt.errorbar(-1*projIggAntPC.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(-1*projIggAntPC.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(-1*projIggAntPC.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-1, 0, 1, 2, 3, 4], [-1, 0, 1, 2, 3, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print('pearson correlation and p-value for Igg Psy PC')
print(sc.stats.pearsonr(projIggPsyPC.iloc[0:41,0], igg_binding.iloc[0:41,2].values))

#%% Correlation w/ Novel clones, with blosum filtering
plt.figure()
plt.errorbar(projIggAntUR.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(projIggAntUR.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(projIggAntUR.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(projIggAntUR.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)
print('pearson correlation and p-value for out of library, blosum filtered Ant UR')
print(sc.stats.pearsonr(projIggAntUR.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding']))


plt.figure()
plt.errorbar(projIggPsyUR.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(projIggPsyUR.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(projIggPsyUR.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(projIggPsyUR.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0.2, 0.6, 1.0, 1.4], [0.2, 0.6, 1.0, 1.4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)
print('pearson correlation and p-value for out of library, blosum filtered Psy UR')
print(sc.stats.pearsonr(projIggPsyUR.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding']))


# physicochemical features
plt.figure()
plt.errorbar(-1*projIggAntPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(-1*projIggAntPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(-1*projIggAntPC.iloc[8,0], 1.2, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(-1*projIggAntPC.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([-1, 0, 1, 2, 3,4], [-1, 0, 1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)
antCorr = sc.stats.mstats.spearmanr(projIggAntPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding'], use_ties = 'True')
print('pearson correlation and p-value for out of library, blosum filtered Ant PC')
print(sc.stats.pearsonr(projIggAntPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT Binding']))


plt.figure()
plt.errorbar(-1*projIggPsyPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], yerr = igg_binding.loc[igg_binding['Blosum62'] == 1,'ANT STDEV'], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(-1*projIggPsyPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(-1*projIggPsyPC.iloc[8,0], 0.51, c = 'orange', s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)
plt.scatter(-1*projIggPsyPC.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([2,6,10,14], [2,6,10,14], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)
print('pearson correlation and p-value for out of library, blosum filtered Psy PC')
print(sc.stats.pearsonr(projIggPsyPC.loc[igg_binding['Blosum62'] == 1,0], igg_binding.loc[igg_binding['Blosum62'] == 1,'OVA Binding']))



