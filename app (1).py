import streamlit as st
import pickle

st.title("Battle of texts: AI vs Human Text Detection")
st.subheader("Enter the text")

input_text = st.text_input("")

# Load the saved pipeline
with open('text_classification_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Transform and predict using the pipeline
if input_text:
    # Transform the input text and make prediction
    prediction_result = pipeline.predict([input_text])
    
    # Display the prediction result
    st.write("Prediction result:", prediction_result)
