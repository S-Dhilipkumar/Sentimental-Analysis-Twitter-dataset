import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("D:\\Sentimental Analysis\\best_sentiment_model.pkl")
vectorizer = joblib.load("D:\\Sentimental Analysis\\tfidf_vectorizer.pkl")

# App title
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Real-Time Sentiment Analyzer")
st.write("Enter a tweet or sentence to predict its sentiment.")

# User input
text_input = st.text_area("✏️ Enter your text here", height=150)

# Predict sentiment
if st.button("Analyze Sentiment"):
    if text_input.strip():
        text_vector = vectorizer.transform([text_input])
        prediction = model.predict(text_vector)[0]
        st.success(f"🔍 Sentiment: **{prediction.upper()}**")
    else:
        st.warning("⚠️ Please enter some text.")
