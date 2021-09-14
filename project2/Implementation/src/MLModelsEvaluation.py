#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def computeAccuracy(myY_true, myY_pred):
    return metrics.accuracy_score(y_true = myY_true, y_pred = myY_pred)

def showMetrics(myTitle, myY_true, myY_pred):
    confusion_matrix =  pd.crosstab(index=np.ravel(myY_true), columns=myY_pred.ravel(), rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, square=False, fmt='', cbar=False)
    plt.title(myTitle, fontsize = 15)
    plt.show()
    print(metrics.classification_report(y_true = myY_true, y_pred = myY_pred))

def compareModels(myTitle,model, X_train, y_train, X_val, y_val):
    model.fit(X_train, np.ravel(y_train))
    predicted_train = model.predict(X_train)
    predicted = model.predict(X_val)
    confusion_matrix =  pd.crosstab(index=np.ravel(y_val), columns=predicted.ravel(), rownames=['Expected'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, square=False, fmt='', cbar=False)
    plt.title(myTitle + "  " , fontsize = 15)
    plt.show()
    print (metrics.classification_report(y_val,predicted))
    print("accuracy Train: ", np.round(metrics.accuracy_score(y_train , predicted_train),3), "\n")
    print("accuracy Validation: ", np.round(metrics.accuracy_score(y_val , predicted),3), "\n")
    
def plotValues(myTitle, myLabelX, myScores, myXList):
    plt.figure(figsize=(10,5))
    plt.title(myTitle)
    plt.plot(myScores, 'go--',label = 'validation')
    plt.xticks(np.arange(len(myXList)), myXList, rotation = 45)
    plt.xlabel(myLabelX)
    plt.ylabel('Accuracy Scores')
    plt.legend()
    plt.show() 

