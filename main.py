import glob
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
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

# Modell evaluieren
accuracy = model_evaluation(X_test_tfidf, y_test, model)

# Ergebnis ausgeben
print(f"Precision: {accuracy:.3f}")


# Beispieltext, den du vorhersagen möchtest
new_text = """
    Pakistan and Sweden have signed a
commodity exchange agreement for 88 mln dlrs each way, the
Pakistan government announced.
    Pakistan's exports under the agreement will include raw
cotton, cotton products, cotton textiles, steel products,
molasses, naphtha and fresh and dried fruits.
    Swedish exports to Pakistan will include medical and
laboratory equipment, electrical telecommunication equipment,
diesel engine spares, mining and security equipment,
road-building and construction machinery, fertilisers and palm
oil.
    """

# Schritt 1: Text in denselben numerischen Vektor umwandeln, der für das Training verwendet wurde
new_text_tfidf = vectorizer.transform([new_text])

# Schritt 2: Vorhersage des Themas mit dem trainierten Modell
predicted_topic = model.predict(new_text_tfidf)

# Ausgabe des vorhergesagten Themas
print(f"Vorhergesagtes Thema: {predicted_topic[0]}")
