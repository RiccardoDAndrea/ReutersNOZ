import glob
import pandas as pd
import re
from bs4 import BeautifulSoup

# Funktion zum Extrahieren von Artikeln + Topics + Train/Test-Split
def extract_articles_with_topics_and_split(file_path):
    """
    Extracts HTML code from all files according to the following criteria: 
    body, topic, lewissplit.
    Regex removes all special characters to get raw text 
    Saved in a dataframe for further processing
    
    file_path:
    ---
    Path to files
    """
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        soup = BeautifulSoup(file.read(), "html.parser")

        articles_data = []
        
        for reuters in soup.find_all("reuters"):
            # Body-Text extrahieren
            body = reuters.body.get_text(strip=True) if reuters.body else ""
            text = body.strip()

            # Nur Topics aus <TOPICS><D>...</D></TOPICS> extrahieren
            topics_tag = reuters.find("topics")
            topics = [topic.get_text(strip=True) for topic in topics_tag.find_all("d")] if topics_tag else []
            topics = ",".join(topics)  # Liste zu String machen

            # Train/Test-Split aus Attribut `LEWISSPLIT`
            split = reuters.get("lewissplit", "").upper()  # Falls Attribut fehlt, leere Zeichenkette zurückgeben

            # HTML-Entities entfernen
            text = re.sub(r'&#\d+;', '', text)

            if text:  # Nur hinzufügen, wenn Text existiert
                articles_data.append({"topics": topics, "text": text, "split": split})

        return articles_data
    
#------------------------------------------------------------------------------------------------------------------------------------------------

def filter_data(dataframe):
    """
    desc
    """
    train_df = dataframe[dataframe['split'] == "TRAIN"]
    test_df = dataframe[dataframe['split'] == "TEST"]
    return train_df, test_df

df = pd.read_csv("reuters_articles_with_split.csv")
df = df.dropna()

train_df, test_df = filter_data(df)


def train_split(dataframe):
    """
    desc
    """
    X_train = dataframe.drop(columns=["topics","split"])
    y_train = dataframe["topics"]

    return X_train, y_train

X_train, y_train = train_split(train_df)

print(X_train.shape, y_train.shape)

def test_split(dataframe):
    """
    desc
    """
    X_test = dataframe.drop(columns=["topics","split"])
    y_test = dataframe["topics"]

    return X_test, y_test

X_test, y_test = test_split(train_df)
print(X_test.shape, y_test.shape)
