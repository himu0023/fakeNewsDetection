# Fake News Detection using machine learning

This project implements a Fake News Detection system using traditionla Machine Learning and Natural Language Processing (NLP). It is trained on the ISOT Fake News Dataset, which contains real news from Reuters and fake news from various unrealiable websites.

A trained Linear Support Vector Machine (SVM) model is deployed using a Streamlit web application that allows users to classify news articles as Real or Fake and perform batch predictions using files.

## Important Disclaimer

This system does NOT verify factual truth.

It learns writing and liggustic patters of:
- Real news (Reuters)
- Fake news (flagged unreliable website)

  Therefore, it detects source style and text patterns, not real-world truth.
  It should not be used as a real fact-checking tool.

  ## Project Structure

fakeNewsDetection/
│
├── app.py
├── models/
│ └── best_model.pkl
├── data/
│ └── True.csv
│ └── Fake.csv
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_modeling.ipynb
├── src/
│ ├── init.py
│ └── text_preprocessing.py
├── requirements.txt
└── setup.py

## Dataset 

ISOT Fake News Dataset

- Real news collected from Reuters
- Fake news collected from PolitiFact-flagged and Wikipedia flagged websites
- Data range: 2016-17
- Columns:
    - title
    - text
    - subject
    - date
Download from Kaggle:
 https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Place `True.csv` and `Fake.csv` inside the `data/` folder.

## Machine learning Pipeline

Preprocessing:
- Lowercasing
- URL removal
- HTML removal
- Punctuation removal
- Stopword removal

Feature Extraction:
- TF-IDF
- Unigrams and Bigrams
- Sublinear term frequency

Model tested:
- Naive Bayes
- Logistic Regression
- Linear SVM

Best Model:
Linear SVM with No-stopwords preprocessing 
F1-score ≈ 0.998

## Streamlit app allows:
- Single article classification
- Batch CSV upload
- Downloading prediction result

  Run the APP:
  -  pip install -r requirements.txt
  -  pip install -e .
  -  streamlit run app.py
 
 ## 📄 CSV Format for Batch Upload

Your CSV must contain a column with article text.

Example:

| article |
|--------|
| Biden signs new bill today... |
| Breaking: shocking discovery... |


##  Technologies Used

- Python  
- Scikit-learn  
- NLTK  
- Pandas  
- Streamlit  
- Joblib  


##  What This Project Demonstrates

- NLP preprocessing  
- TF-IDF feature extraction  
- Model comparison and tuning  
- Fake-news classification  
- Deployment with Streamlit  
- Batch predictions  


##  License

This project is for educational and research purposes only
