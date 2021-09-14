#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#Generate counter for words and extract features
def countVectorizer(corpus):
    sentences = corpus
    vectorizer = CountVectorizer(analyzer= 'word')
    sentence_vectors = vectorizer.fit_transform(sentences)
    vector_arr = sentence_vectors.toarray()
    features_name = vectorizer.get_feature_names()
    return vector_arr, features_name

#create target label (0/1)
def createTrainSetandLabel(good_instr, bad_instr):
    corpus_combined = []
    id_combined =[]
    ops_combined = []
    target_combined = []
    cnt_good =0
    cnt_bad = 0
    for g in good_instr:
        if len(g)<2:
            continue
        else:
            id_combined.append(g[0])
            corpus_combined.append(g[1])
            target_combined.append(0)
            cnt_good += 1
    for b in bad_instr:
        if len(b)<2:
            continue
        else:
            id_combined.append(b[0])
            corpus_combined.append(b[1])
            target_combined.append(1)
            cnt_bad += 1
    print("good functions: ", cnt_good)
    print("bad functions: ", cnt_bad)
    return id_combined, corpus_combined, target_combined

#Export Train dataset to excel (used for fitting models)
def exportToExcel(filename, features_name, counter_arr, id_combined, target_combined):
    df=pd.DataFrame.from_items(zip(features_name, counter_arr.T))
    df['target'] = target_combined
    df['fname'] = id_combined
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()

