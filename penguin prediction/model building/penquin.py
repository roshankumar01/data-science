# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 18:36:42 2021

@author: Roshan
"""


import pandas as pd
df = pd.read_csv("penquin.csv")

df = df.drop(columns="Unnamed: 0")
df.columns

target = ["species"]
catogorical_var = ["sex","island"]

df_1 = df.copy()

for col in catogorical_var:
    df_dummy = pd.get_dummies(df[col])
    df_1 = pd.concat([df_1,df_dummy],axis=1)
    df_1.drop(col,axis=1,inplace=True)
    
df[df.columns[0]].unique()
target_mapper = {"Adelie":0,"Gentoo":1,"Chinstrap":2}



df_1['species']=df_1['species'].map(target_mapper)

y = df_1.pop('species')
x = df_1

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.9)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

print("training  acc",accuracy_score(y_train,rf.predict(x_train)))

print("test  acc",accuracy_score(y_test,rf.predict(x_test)))

print("train matrix" ,confusion_matrix(y_train,rf.predict(x_train)))
print("test matrix" ,confusion_matrix(y_test,rf.predict(x_test)))

import pickle
pickle.dump(rf, open("penguin_clf.pkl","wb"))
