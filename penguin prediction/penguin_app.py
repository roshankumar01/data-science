# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:54:48 2021

@author: Roshan
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
        ## penguin app
         """)
         
st.sidebar.header("user input")

#st.sidebar.markdown("""
#    [example csv](url)
# """)

upload_file = st.sidebar.file_uploader("input file",type=["csv"])
#df.columns

if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_function():
        island = st.sidebar.selectbox("Island",('Torgersen', 'Biscoe', 'Dream'))
        sex = st.sidebar.selectbox("sex" ,('male','female'))
        bill_length_mm = st.sidebar.slider("bill length",32.1,59.6,40.1)
        bill_depth_mm = st.sidebar.slider("bill depth",13.1,21.5,16.3)
        flipper_length_mm =  st.sidebar.slider("flipper length",172,231,200)
        body_mass_g = st.sidebar.slider("body mass",2700,6300,3500)
        
        data = {'island':island,
                'bill_length_mm':bill_length_mm,
                'bill_depth_mm':bill_depth_mm,
                'flipper_length_mm':flipper_length_mm,
                'body_mass_g':body_mass_g,
                'sex':sex
                }
        features = pd.DataFrame(data,index=[0])
        return features
    input_df=user_input_function()        
    
penguin_raw = pd.read_csv("penquin.csv")
penguin_raw = penguin_raw.drop(columns="Unnamed: 0")
penguin = penguin_raw.drop(columns = ['species'])

df = pd.concat([input_df,penguin],axis=0)

cat =['sex','island']
for col in cat:
    dummy = pd.get_dummies(df[col])
    df=pd.concat([df,dummy],axis=1)
    df=df.drop(columns=[col])
    
df = df[:1]

st.subheader("input features")

if upload_file is not None:
    st.write(df)
else:
    st.write("waiting for csv , currently using example parameters")
    st.write(df) 

load_clf = pickle.load(open("penguin_clf.pkl","rb"))  

pred = load_clf.predict(df)  
pred_proba = load_clf.predict_proba(df) 

st.subheader("Prediction")
pen_spec = np.array(['Adelie','Gentoo','Chinstrap'])
st.write(pen_spec[pred])

st.write('prediction probability')
st.write(pred_proba)