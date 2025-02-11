import glob,re
import pandas as pd 
from bs4 import BeautifulSoup
import pandas as pd

# Function to import
from functions import extract_articles_with_topics_and_split, all_articles, create_df, data_preprocessing, filter_data, train_split, test_split,vectorize, create_model, model_evaluation

articles_list = all_articles(dir="reuters21578")  
df = create_df(articles_list)
df = data_preprocessing(df)

# # Prüfen, ob DataFrame gefüllt wurde
# if df.empty:
#     print("DataFrame empty.")
# else:
#     print("DataFrame created.")

# Train/Test-Split durchführen
train_df, test_df = filter_data(df)

# Trainings- und Testdaten vorbereiten
X_train, y_train = train_split(train_df)
X_test, y_test = test_split(test_df)

# TF-IDF Vektorisierung
X_train_tfidf, X_test_tfidf, vectorizer = vectorize(X_train, X_test)

# Modell trainieren
model = create_model(X_train_tfidf, y_train)

accuracy = model_evaluation(X_test_tfidf, y_test, model)
print(f"Accuracy: {accuracy:.3f}")


# Beispieltext, der vorhergesagt werden soll
new_text = """
    Shr loss nil vs profit 19 cts
    Net loss 3,175 vs profit 284,945
    Revs 13.6 mln vs 10.6 mln
    Year
    Shr profit 13 cts vs profit 56 cts
    Net profit 195,202 vs profit 857,006
    Revs 47.5 mln vs 42.9 mln
    Note: Current year net includes charge against discontinued
operations of 1,060,848 dlrs.
    """

# Schritt 1: Text in denselben numerischen Vektor umwandeln, der für das Training verwendet wurde
new_text_tfidf = vectorizer.transform([new_text])

# Schritt 2: Vorhersage des Themas mit dem trainierten Modell
predicted_topic = model.predict(new_text_tfidf)

# Ausgabe des vorhergesagten Themas
print(f"Vorhergesagtes Thema: {predicted_topic[0]}")
