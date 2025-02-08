import glob
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Funktion zum Extrahieren von Artikeln + Topics + Train/Test-Split
def extract_articles_with_topics_and_split(file_path):
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        soup = BeautifulSoup(file.read(), "html.parser")

        articles_data = []
        
        for reuters in soup.find_all("reuters"):
            body = reuters.body.get_text(strip=True) if reuters.body else ""
            text = body.strip()

            topics_tag = reuters.find("topics")
            topics = [topic.get_text(strip=True) for topic in topics_tag.find_all("d")] if topics_tag else []
            topics = ",".join(topics)

            split = reuters.get("lewissplit", "").upper() 

            text = re.sub(r'&#\d+;', '', text)

            if text:  
                articles_data.append({"topics": topics, "text": text, "split": split})

        return articles_data

# Alle Artikel laden
def all_articles(dir: str):
    sgm_files = glob.glob(f"{dir}/*.sgm")
    all_articles = []

    for file_path in sgm_files:
        articles = extract_articles_with_topics_and_split(file_path)
        all_articles.extend(articles)

    return all_articles

# DataFrame erstellen
def create_df(articles_list):
    return pd.DataFrame(articles_list)

# Fehlende Werte entfernen
def data_preprocessing(dataframe):
    return dataframe.dropna()

# Train/Test-Split
def filter_data(dataframe):
    train_df = dataframe[dataframe['split'] == "TRAIN"]
    test_df = dataframe[dataframe['split'] == "TEST"]
    return train_df, test_df

# Trainingsdaten vorbereiten
def train_split(dataframe):
    X_train = dataframe["text"]  # Nur den Text verwenden!
    y_train = dataframe["topics"]
    return X_train, y_train

# Testdaten vorbereiten
def test_split(dataframe):
    X_test = dataframe["text"]  # Nur den Text verwenden!
    y_test = dataframe["topics"]
    return X_test, y_test

# TF-IDF-Vektorisierung
def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') 
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer  # Vectorizer wird zurückgegeben!

# Modell trainieren
def create_model(X_train_tfidf, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model

# Modell evaluieren
def model_evaluation(X_test_tfidf, y_test, model):
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy



