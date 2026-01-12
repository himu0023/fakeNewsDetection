import streamlit as st 
import joblib
import numpy as np 
import pandas as pd 
from src.text_preprocessing import clean_no_stopwords

# Load trained Pipeline
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Detection Demo")
st.write(
    "This app uses a machine learning model trained on the ISOT Fake News Dataset"
    "(Reuters vs flagged fake news sites). It detects **writing patterns**, not truth."
    )

st.markdown("------")

# SINGLE TEXT PREDICTION

st.subheader("Single Article")

user_text = st.text_area("Paste a nes article or paragraph: ")

if st.button("Analyze Single Article"):
    if len(user_text.strip()) < 50:
        st.warning("Please enter a longer news article for meaningful prediction.")
    else:
        pred = model.predict([user_text])[0]
        score = model.decision_function([user_text])[0]
        confidence = 1/(1 + np.exp(-abs(score)))

        if pred == 1:
            st.success("🟢 Classified as REAL news")
        else: 
            st.error("🔴 Classified as FAKE news")

        st.write(f"Model confidence (approx): **{confidence:.2f}**")


st.markdown("------")


# BATCH CVS PREDICTION 

st.subheader("Batch CSV Upload")

upload_file = st.file_uploader("Upload a CSV file containing news articles", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.write("Preview of uploaded data: ")
    st.dataframe(df.head())

    text_column = st.selectbox("Select the column that contains the article text: ", df.columns)

    if st.button("Analyze CSV"):
        texts = df["text_column"].astype(str).tolist()

        preds = model.predict(texts)
        scores = model.decision_function(texts)

        # Convert Predictions
        labels = ["Real" if p ==1 else "Fake" for p in preds]
        confidence = 1/(1+np.exp(-np.abs(scores)))

        df["Prediction"] = labels
        df["Confidence"] = confidence

        st.write("Results: ")
        st.dataframe(df.head())

        # Download Button 
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data = csv,
            file_name="fake_news_results.csv",
            mime="text/csv"
        )


st.markdown("------")

st.caption(
    "⚠️ This model learns dataset specific writing patters."
    "It should not used as real-world fact checking system."
)