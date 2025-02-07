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


