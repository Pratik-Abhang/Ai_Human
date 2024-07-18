import streamlit as st
import pickle 

st.title("Battle of texts: AI vs Human Text Detection")
st.subheader("Enter the text")


input_text = st.text_input("")

# Load the saved components
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

with open('truncated_svd.pkl', 'rb') as f:
    loaded_svd = pickle.load(f)

with open('svc.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Function to preprocess text using TF-IDF and SVD
def preprocess_text(text, vectorizer, svd):
    # Vectorize input text using TF-IDF
    text_tfidf = vectorizer.transform([text])

    # Reduce dimensionality using SVD
    text_svd = svd.transform(text_tfidf)

    return text_svd

# Preprocess input text and make prediction
if input_text:
    input_svd = preprocess_text(input_text, loaded_tfidf, loaded_svd)
    prediction_result = loaded_model.predict(input_svd)
    
    # Display the prediction result
    st.write("Prediction result:", prediction_result)
