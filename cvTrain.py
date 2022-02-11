# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:39:55 2022

@author: pkinn
"""
def cvTrain(model, features, targets, nSplits, nEpochs, batchSz, initWts):
    from sklearn.model_selection import KFold
    import numpy as np
    kf = KFold(n_splits = nSplits, shuffle = True)
    fn = 1
    # Define per-fold score containers <-- these are new
    acc_per_fold = []
    loss_per_fold = []
    allHist = np.zeros((nSplits, nEpochs))
    for train, test in kf.split(features, targets):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fn} ...')
        model.set_weights(initWts)
        for kk in range(nEpochs):
            history = model.fit(features[train], targets[train], 
                                batch_size = batchSz, 
                                epochs = 1,
                                verbose = 0)
            scores = model.evaluate(features[test], targets[test], verbose = 0)
            allHist[fn-1, kk] = scores[1]
        print(f'Score for fold {fn}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fn += 1
        
    return acc_per_fold, loss_per_fold, allHist