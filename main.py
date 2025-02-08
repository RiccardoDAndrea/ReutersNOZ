import glob
import pandas as pd
import re
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
    Total food aid needs in 69 of the
least developed countries declined in 1986/87, as requirments
fell in many countries in Africa, the Middle East and Asia, the
U.S. Agriculture Department said.
    In a summary of its World Agriculture Report, the
department said grain production in sub-Saharan Africa was a
record high in 1986, with gains in almost every country.
    However, food needs in Central America rose, worsened by
drought-reduced crops and civil strife.
    Record wheat production in 1986/87 is pushing global wheat
consumption for food to a new high, and higher yielding
varieties have been particularly effective where spring wheat
is a common crop, it said.
    However, may developing countries in tropical climates,
such as Sub-Saharan Africa, Southeast Asia, and Central
America, are not well adapted for wheat production, and
improved varieties are not the answer to rising food needs, the
department said.
    World per capita consumption of vegetable oil will rise in
1986/87 for the third straight year.
    Soybean oil constitutes almost 30 pct of vegetable oil
consumption, while palm oil is the most traded, the department
said.
    """

# Schritt 1: Text in denselben numerischen Vektor umwandeln, der für das Training verwendet wurde
new_text_tfidf = vectorizer.transform([new_text])

# Schritt 2: Vorhersage des Themas mit dem trainierten Modell
predicted_topic = model.predict(new_text_tfidf)

# Ausgabe des vorhergesagten Themas
print(f"Vorhergesagtes Thema: {predicted_topic[0]}")
