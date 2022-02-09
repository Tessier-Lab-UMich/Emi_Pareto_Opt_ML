# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:16:30 2022

@author: pkinn
"""
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
import numpy as np
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


#%% Do KFold and record results

from cvTrain import cvTrain
targetsAnt = emi_binding['ANT Binding'].values
targetsPsy = emi_binding['OVA Binding'].values
nEpoch = 500
batch_sz = 50
pdAntOH = projectorDecider(1, emi_onehot.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdAntOH.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
initWeights = pdAntOH.get_weights()
[acc, loss, pdAntOHhist] = cvTrain(pdAntOH, emi_onehot.values, targetsAnt, 5, nEpoch, batch_sz, initWeights)

pdPsyOH = projectorDecider(1, emi_onehot.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdPsyOH.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
initWeights = pdPsyOH.get_weights()
[acc, loss, pdPsyOHhist] = cvTrain(pdPsyOH, emi_onehot.values, targetsPsy, 5, nEpoch, batch_sz, initWeights)

pdAntUR = projectorDecider(1, emi_reps.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdAntUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
initWeights = pdAntUR.get_weights()
[acc, loss, pdAntURhist] = cvTrain(pdAntUR, emi_reps.values, targetsAnt, 5, nEpoch, batch_sz, initWeights)

pdPsyUR = projectorDecider(1, emi_reps.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdPsyUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
initWeights = pdPsyUR.get_weights()
[acc, loss, pdPsyURhist] = cvTrain(pdPsyUR, emi_reps.values, targetsPsy, 5, nEpoch, batch_sz, initWeights)

pdAntPC = projectorDecider(1, emi_physvh.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdAntPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
initWeights = pdAntPC.get_weights()
[acc, loss, pdAntPChist] = cvTrain(pdAntPC, emi_physvh.values, targetsAnt, 5, nEpoch, batch_sz, initWeights)

pdPsyPC = projectorDecider(1, emi_physvh.shape[1])
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
pdPsyPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
initWeights = pdPsyPC.get_weights()
[acc, loss, pdPsyPChist] = cvTrain(pdPsyPC, emi_physvh.values, targetsPsy, 5, nEpoch, batch_sz, initWeights)

allHist = [pdAntOHhist, pdPsyOHhist, pdAntURhist, pdPsyURhist, pdAntPChist, pdPsyPChist]
allNames = ['Ant/OH', 'Psy/OH', 'Ant/UR', 'Psy/UR', 'Ant/PC', 'Psy/PC']


#%% plot results
plt.figure()
for ii in range(len(allHist)):
    plt.subplot(3,2,ii+1)
    plt.plot(allHist[ii].transpose())
    plt.plot(np.mean(allHist[ii].transpose(), 1), 'k')
    plt.title(allNames[ii])
    
        
        
