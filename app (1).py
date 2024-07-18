import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import math
import streamlit as st
from sklearn.model_selection import train_test_split
import pickle




st.title("AI vs human")
st.subheader("Enter the text")

input_text = st.text_input("")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
input_text = vectorizer.fit_transform(['input_text'])
 

#dt=pd.read_csv("AI_Human.csv")
#st.write(dt.head())

# dt["len"]=dt["text"].apply(len)
# ai=dt[dt['generated']==1]
# human=dt[dt['generated']==0]

# dt['generated_cor'] = dt['generated'].replace({
#     1: 'AI',
#     0: 'Human'
# })

#spliting data

#X_train,x_test,y_train,y_test=train_test_split(dt['text'],dt['generated_cor'],test_size=0.2,random_state=42)

# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB, ComplementNB
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import learning_curve
# from sklearn.linear_model import LogisticRegression

#load the model

with open('model.pkl','rb') as f:
    pipe_logistic=pickle.load(f)

def predict():
    predict_logistic = pipe_logistic.predict([input_text])
    return predict_logistic

if st.button('Predict'):
    prediction_result = predict()
    st.write(prediction_result)












