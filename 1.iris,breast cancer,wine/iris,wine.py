import streamlit as st
import sklearn
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

st.title('example')

st.write("""
         # Explore different classifier
         which one is best
    """)

dataset_name = st.sidebar.selectbox("SELECT DATASET", ("Iris","Breast cancer","Wine"))

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random forest"))

def get_data(dataset_name):
    if dataset_name=="Iris":
        data = datasets.load_iris()
    if dataset_name=="Breast cancer":
        data = datasets.load_breast_cancer()
    if dataset_name=="Wine":
        data = datasets.load_wine()
    x=data.data
    y=data.target
    return x,y

x,y = get_data(dataset_name)
st.write("shape of dataset ",x.shape)   
st.write("no of classes ", len(np.unique(y)))


def add_parameter(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params['K'] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params['C'] = C
    else:
       max_depth = st.sidebar.slider("max_depth",2,15)
       n_est = st.sidebar.slider("n_estimator",1,100)
       params['max_depth'] = max_depth
       params['n_estimators'] = n_est
      
    return params


params = add_parameter(classifier_name)  

def classifier(classifier_name,params):
    
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    elif classifier_name=="SVM":
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'])
    
    return clf 

clf = classifier(classifier_name,params)
 
# classification

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc= accuracy_score(y_test,y_pred)

st.write("classifier =",classifier_name)
st.write("accurracy =", acc)


#plot

pca =PCA(2)

x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]
 

fig = plt.figure()
        
plt.scatter(x1,x2,cmap='viridis',c=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.colorbar()

st.pyplot(fig)























