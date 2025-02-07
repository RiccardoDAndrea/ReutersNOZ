import glob
import pandas as pd
import re
from bs4 import BeautifulSoup
from functions import extract_articles_with_topics_and_split



# Alle SGM-Dateien finden
sgm_files = glob.glob("reuters21578/*.sgm")

# Daten aus allen Dateien extrahieren
all_articles = []

for file_path in sgm_files:
    articles = extract_articles_with_topics_and_split(file_path)
    all_articles.extend(articles)

# DataFrame erstellen
df = pd.DataFrame(all_articles)

# Ergebnis anzeigen

# DataFrame als CSV speichern
df.to_csv("reuters_articles_with_split.csv", index=False)
print(df.head(20))