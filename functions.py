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
    """
    Extracts data from the Reuters dataset using BeautifulSoup.

    Retrieves:
        - split (lewissplit): Article splitting information.
        - text: Cleaned article text.
        - topics: Comma-separated string of relevant topics.

    Parameters
    ----------
    file_path : str
        Path to the HTML file containing the Reuters data.

    Returns
    -------
    List of JSON objects with keys 'topics', 'text', and 'split'.
    """
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
    """
    Collects and returns all article data from SGML files in the specified directory.

    Parameters
    ----------
    dir : str
        Path to the directory containing the SGML files.

    Returns
    -------
    list of JSON objects with keys 'topics', 'text', and 'split'
    """
    sgm_files = glob.glob(f"{dir}/*.sgm")
    all_articles = []

    for file_path in sgm_files:
        articles = extract_articles_with_topics_and_split(file_path)
        all_articles.extend(articles)

    return all_articles

# DataFrame erstellen
def create_df(articles_list):
    """
    Converts a list of JSON objects into a pandas DataFrame.

    Parameters
    ----------
    articles_list : list
        List of dictionaries with 'topics', 'text', and 'split' keys.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the article data.
    """

    return pd.DataFrame(articles_list)

# Fehlende Werte entfernen
def data_preprocessing(dataframe):
    """
    Performs basic data preprocessing on a DataFrame, dropping all rows with missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame to be preprocessed.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame without any NaN values.
    """
    return dataframe.dropna()

# Train/Test-Split
def filter_data(dataframe):
    """
    Splits a DataFrame into two DataFrames based on the 'split' column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame to be filtered.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing two DataFrames: `train_df` and `test_df`.
    """
    train_df = dataframe[dataframe['split'] == "TRAIN"]
    test_df = dataframe[dataframe['split'] == "TEST"]
    return train_df, test_df

# Trainingsdaten vorbereiten
def train_split(dataframe):
    """
    Splits the data into dependent and independent variables

    Parameters
    ----------
    dataframe : dataframe
            Dataframe object
    Returns:
    ---
    Two dataset
    """
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



