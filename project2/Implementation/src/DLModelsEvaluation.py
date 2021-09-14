#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# To compute accuracy
def computeAccuracy(myY_true, myY_pred):
    return metrics.accuracy_score(y_true = myY_true, y_pred = myY_pred)

# To create classification metric 
def showMetrics(myTitle, myY_true, myY_pred):
    confusion_matrix =  pd.crosstab(index=np.ravel(myY_true), columns=myY_pred.ravel(), rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, square=False, fmt='', cbar=False)
    plt.title(myTitle, fontsize = 15)
    plt.show()
    print(metrics.classification_report(y_true = myY_true, y_pred = myY_pred))
    
def plotValues(myTitle, myLabelX, myScores, myXList):
    plt.figure(figsize=(10,5))
    plt.title(myTitle)
    plt.plot(myScores, 'go--',label = 'validation')
    plt.xticks(np.arange(len(myXList)), myXList, rotation = 45)
    plt.xlabel(myLabelX)
    plt.ylabel('Accuracy Scores')
    plt.legend()
    plt.show() 

def plotLoss(history, ymin, ymax):
    colors = ["#4374B3","#18A614" ]
    sns.set_palette(sns.color_palette(colors))
    plt.plot(history.history['loss'], label='train_loss') # For TF2
    plt.plot(history.history['val_loss'], label = 'valid_loss') # For TF2
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim([ymin, ymax])
    plt.legend(loc='lower right')

def plotAccuracy(history, ymin, ymax):
    colors = ["#4374B3","#18A614" ]
    sns.set_palette(sns.color_palette(colors))
    plt.plot(history.history['accuracy'], label='train_accuracy') # For TF2
    plt.plot(history.history['val_accuracy'], label = 'valid_accuracy') # For TF2
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([ymin, ymax])
    plt.legend(loc='lower right')
