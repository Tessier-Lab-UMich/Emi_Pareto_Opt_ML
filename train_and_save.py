# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:36:22 2022

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

#%% Train "final" models
# Set up training parameters
targetsAnt = emi_binding['ANT Binding'].values
targetsPsy = emi_binding['OVA Binding'].values
batch_sz = 50
nepochs = 50
nepochsUR = 250
nIntermedNodes = 20
savepath = f'deepNNWeights{os.sep}node{nIntermedNodes}{os.sep}'
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

#%% Train and save weights
pdAntOH = deepProjectorDecider(1, emi_onehot.shape[1], nIntermedNodes)
pdAntOH.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntOH.fit(emi_onehot.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)
wtName = 'pdAntOH_wts'
pdAntOH.save_weights(savepath+wtName)




pdPsyOH = deepProjectorDecider(1, emi_onehot.shape[1], nIntermedNodes)
pdPsyOH.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyOH.fit(emi_onehot.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)
wtName = 'pdPsyOH_wts'
pdPsyOH.save_weights(savepath+wtName)




pdAntUR = deepProjectorDecider(1, emi_reps.shape[1], nIntermedNodes)
pdAntUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntUR.fit(emi_reps.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = nepochsUR,
                        verbose = 1)
wtName = 'pdAntUR_wts'
pdAntUR.save_weights(savepath+wtName)




pdPsyUR = deepProjectorDecider(1, emi_reps.shape[1], nIntermedNodes)
pdPsyUR.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyUR.fit(emi_reps.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochsUR,
                        verbose = 1)
wtName = 'pdPsyUR_wts'
pdPsyUR.save_weights(savepath+wtName)



pdAntPC = deepProjectorDecider(1, emi_physvh.shape[1], nIntermedNodes)
pdAntPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdAntPC.fit(emi_physvh.values, targetsAnt, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)
wtName = 'pdAntPC_wts'
pdAntPC.save_weights(savepath+wtName)



pdPsyPC = deepProjectorDecider(1, emi_physvh.shape[1], nIntermedNodes)
pdPsyPC.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
pdPsyPC.fit(emi_physvh.values, targetsPsy, 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)
wtName = 'pdPsyPC_wts'
pdPsyPC.save_weights(savepath+wtName)

