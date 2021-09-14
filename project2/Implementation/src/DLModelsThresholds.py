#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#To predict the class label (0/1)  
def getPredictedLabel(dl_outputs, threshold):
    pred_labels = []
    for output in dl_outputs:
        if output[0] <= threshold:
            pred_labels.append(0)
        else:
            pred_labels.append(1)
    return np.array(pred_labels)  

#To compute accuracy
def computeAccuracy(myY_true, myY_pred):
    return metrics.accuracy_score(y_true = myY_true, y_pred = myY_pred)

#To compute F1
def computeF1(myY_true, myY_pred):
    return metrics.f1_score(y_true = myY_true, y_pred = myY_pred)


#To plot the metrics across threshold
def plotAccuracyandF1(threshold_list, y_true,preds):
    acc_list = []
    f1_list = []
    for t in threshold_list:
        predicted_class  = getPredictedLabel(preds, t)
        acc_list.append(computeAccuracy(y_true, predicted_class))
        f1_list.append(computeF1(y_true, predicted_class))
    plt.plot(acc_list,  'bo--',label='Accuracy') # For TF2
    plt.plot(f1_list, 'co--', label = 'F1 scores') # For TF2
    plt.xticks(np.arange(len(threshold_list)), threshold_list, rotation = 45)
    plt.xlabel('Threshold')
    plt.ylabel('Metrics')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()