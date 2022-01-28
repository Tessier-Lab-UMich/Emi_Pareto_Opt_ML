# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 20:46:14 2022

@author: pkinn
"""

#%% Imports
import tensorflow as tf
from holdout_utils import *
from onehot_gen import onehot_gen, shiftedColorMap
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns


#%% Load data into dataframes
emi_binding = pd.read_csv("emi_binding.csv", header = 0, index_col = 0)
iso_binding = pd.read_csv("iso_binding.csv", header = 0, index_col = 0)
igg_binding = pd.read_csv("igg_binding.csv", header = 0, index_col = 0)

#%% generate OHE features
emi_onehot = onehot_gen(emi_binding)
iso_onehot = onehot_gen(iso_binding)
igg_onehot = onehot_gen(igg_binding)

#%% Create dense model

modelCompression = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(2)])

modelNormal = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(2)])

lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

model.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])

model.fit(emi_onehot.values,  emi_binding['ANT Binding'].values, epochs = 10)

#%% Use k-fold CV
from sklearn.model_selection import KFold

kf = KFold(n_splits = 10, shuffle = True)
fn = 1
features = emi_onehot.values
batch_sz = 50
nepochs = 10
targets = emi_binding['ANT Binding'].values
# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []
for train, test in kf.split(features, targets):
    
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(1, activation = 'relu'),
        # tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(2)])
    
    lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    
    model.compile(optimizer = 'adam', loss = lossFn, metrics = ['accuracy'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fn} ...')
    
    history = model.fit(features[train], targets[train], 
                        batch_size = batch_sz, 
                        epochs = nepochs,
                        verbose = 1)
    scores = model.evaluate(features[test], targets[test], verbose = 0)
    print(f'Score for fold {fn}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fn += 1


