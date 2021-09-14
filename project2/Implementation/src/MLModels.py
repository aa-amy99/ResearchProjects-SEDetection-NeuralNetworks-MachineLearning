#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


#Build DT from Gridsearch CV
def buildDecisionTreeGS(random_seed, X_train, y_train, MS_list, MD_list, CW_list, scoring = "balanced_accuracy" ):
    params = {'min_samples_split': MS_list,
         'max_depth': MD_list, 'random_state': [random_seed],"class_weight": CW_list}
    dt = DecisionTreeClassifier()
    grid = GridSearchCV(estimator=dt, param_grid=params, verbose=1, cv=5,scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

#Build RF from Gridsearch CV
def buildRandomForestGS(random_seed, X_train, y_train, ET_list, MS_list, MD_list, CW_list, scoring = "balanced_accuracy" ):
    params = {'n_estimators': ET_list,'min_samples_split': MS_list,
         'max_depth': MD_list, 'random_state': [random_seed],"class_weight": CW_list}
    rf = RandomForestClassifier()
    grid = GridSearchCV(estimator=rf, param_grid=params, verbose=1, cv=5,scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

#Build SVM from Gridsearch CV
def buildSVM(X_train, y_train, CG_list, CW_list, scoring = "balanced_accuracy" ):
    params = {'C': CG_list,
          'gamma': CG_list,
           'kernel': ['rbf'],
         "class_weight": CW_list}
    svc = SVC()
    grid = GridSearchCV(estimator=svc, param_grid=params, verbose=1, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

def buildCustomSVM(myC, myGamma, myClassWeight):
    svm = SVC(kernel='rbf', C = 100, gamma =0.01, class_weight=myClassWeight)
    return svm
	

#Fit model from Gridsearch CV
def fitModel(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

#Use model to predict class
def predictClass(model,X):
    return model.predict(X)

#Compute features important scores (only for Decision Tree and Random Forest)
def getImportantFeatures(rank,X_train, model):
    imp = model.feature_importances_
    feat_importances = pd.Series(imp, index=X_train.columns)
    print("Important Scores: ")
    top_features  = feat_importances.nlargest(rank)
    return top_features

#Plot important features
def plotImportantFeatures(top_features):
    plt.figure(figsize=(10,5))
    top_features.plot(kind='barh')
    plt.title("Ranking the important features")
    plt.show()
    
#Plot Decision Tree    
def plotDecisionTree(X_train,DT_model):
    fn = X_train.columns
    cn = ['non-vul','vul']
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20), dpi=200)
    tree.plot_tree(DT_model,feature_names = fn, class_names=cn, filled = True, fontsize=12);
    
#Plot Accuracy curve
def plotAccuracyCurve(x_axis_label, x_list, acc_train,acc_val):
    plt.figure(figsize=(10,5))
    plt.title("The Accuracy Curve")
    plt.plot(acc_train, 'bo--', label = 'train')
    plt.plot(acc_val,'go--', label = 'valid')
    plt.xticks(np.arange(len(x_list)), x_list, rotation = 45)
    plt.xlabel(x_axis_label)
    plt.ylabel('Accuracy Rate')
    plt.legend()
    plt.show()


