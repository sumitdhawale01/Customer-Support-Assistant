import streamlit as st
import joblib
import pandas as pd

# Load saved models
model = joblib.load("query_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load the dataset
df = pd.read_csv("customer_queries.csv")

# Function to generate response
def generate_response(user_query):
    query_tfidf = vectorizer.transform([user_query])  # Transform input query
    predicted_category_encoded = model.predict(query_tfidf)[0]  # Predict category
    predicted_category = label_encoder.inverse_transform([predicted_category_encoded])[0]  # Decode category

    # Retrieve response
    response = df[df['category'] == predicted_category]['response'].values
    if len(response) > 0:
        return predicted_category, response[0]
    else:
        return predicted_category, "Sorry, I don't have a response for that."

# Streamlit UI
st.title("Customer Query Classifier")
st.write("Enter your query below, and the model will predict its category and provide a response.")

# User input
user_input = st.text_input("Type your query here:")

if st.button("Submit"):
    if user_input:
        category, response = generate_response(user_input)
        st.write(f"### Predicted Category: {category}")
        st.write(f"### Response: {response}")
    else:
        st.warning("Please enter a query.")
