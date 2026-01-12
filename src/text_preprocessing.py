import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_no_stopwords(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])