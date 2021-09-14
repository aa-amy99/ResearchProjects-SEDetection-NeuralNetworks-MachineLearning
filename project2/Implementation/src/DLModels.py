#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras.layers import *
from keras import optimizers 
import numpy as np

def reshape_X(X_train_norm_c, X_val_norm_c, X_test_norm_c):
    X_train_DL = X_train_norm_c.reshape((X_train_norm_c.shape[0], 1, X_train_norm_c.shape[1]))
    X_val_DL = X_val_norm_c.reshape((X_val_norm_c.shape[0], 1, X_val_norm_c.shape[1]))
    X_test_DL = X_test_norm_c.reshape((X_test_norm_c.shape[0], 1, X_test_norm_c.shape[1]))
    return X_train_DL, X_val_DL, X_test_DL

def reshape_Y(y_train_c, y_val_c, y_test_c):
    encoder = LabelEncoder()
    encoder.fit(y_train_c)
    y_train_dl = encoder.transform(y_train_c)
    encoder.fit(y_val_c)
    y_val_dl = encoder.transform(y_val_c)
    encoder.fit(y_test_c)
    y_test_dl = encoder.transform(y_test_c)
    return y_train_dl, y_val_dl, y_test_dl

def buildDLmodel(shape, hidden_act, output_act):
    model = Sequential()
    model.add(Bidirectional(LSTM(256, activation=hidden_act, return_sequences=True, input_shape=(1, shape))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256, activation=hidden_act)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation=output_act))
    return model

def compileDLModel(model, my_leanring_rate):
    myoptimizer = optimizers.Adam(learning_rate=my_leanring_rate)
    model.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])
    return model

def fitDLmodel(model, X_train_DL, y_train_dl, X_val_DL, y_val_dl, mybatch_size, myepochs, myclassweigth, mycallback):
    mymodel = model
    history = mymodel.fit(X_train_DL, y_train_dl, batch_size =mybatch_size,epochs=myepochs, validation_data=(X_val_DL, y_val_dl), class_weight= myclassweigth, callbacks=[mycallback])
    return history, mymodel
    
 
    

