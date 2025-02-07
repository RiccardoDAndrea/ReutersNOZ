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



############### C L A S S I F I C A T I O N ############### 