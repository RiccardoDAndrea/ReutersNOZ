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

# articles_list = all_articles(dir="reuters21578")  
# df = create_df(articles_list)
# df = data_preprocessing(df)

# # # Prüfen, ob DataFrame gefüllt wurde
# # if df.empty:
# #     print("DataFrame empty.")
# # else:
# #     print("DataFrame created.")

# # Train/Test-Split durchführen
# train_df, test_df = filter_data(df)

# # Trainings- und Testdaten vorbereiten
# X_train, y_train = train_split(train_df)
# X_test, y_test = test_split(test_df)

# # TF-IDF Vektorisierung
# X_train_tfidf, X_test_tfidf, vectorizer = vectorize(X_train, X_test)

# # Modell trainieren
# model = create_model(X_train_tfidf, y_train)

# # Modell evaluieren
# accuracy = model_evaluation(X_test_tfidf, y_test, model)

# # Ergebnis ausgeben
# print(f"Precision: {accuracy:.3f}")


# # Beispieltext, den du vorhersagen möchtest
# new_text = """
#     Oper shr 38 cts vs 1.84 dlrs
#     Oper net 973,000 vs 4,497,000
#     Nine mths
#     Oper shr 1.22 dlrs vs 1.31 dlrs
#     Oper net 3,133,000 vs 3,410,000
#     NOTE: Results exclude extraordinary gain from net loss  
#     carryforward of 672,000 dlrs or 27 cts in 1987 3rd qtr, 918,000
#     dlrs 38 cts in 1986 3rd qtr, and 1,071,000 dlrs or 44 cts in
#     1987 nine months. 1986 results include 5.1 mln dlr gain from
#     termination of defined benefit pension plan.
#     """

# # Schritt 1: Text in denselben numerischen Vektor umwandeln, der für das Training verwendet wurde
# new_text_tfidf = vectorizer.transform([new_text])

# # Schritt 2: Vorhersage des Themas mit dem trainierten Modell
# predicted_topic = model.predict(new_text_tfidf)

# # Ausgabe des vorhergesagten Themas
# print(f"Vorhergesagtes Thema: {predicted_topic[0]}")

