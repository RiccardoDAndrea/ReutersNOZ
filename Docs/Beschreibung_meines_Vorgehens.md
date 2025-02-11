# Beschreibung der Vorgehensweise

## Ziel
Das Ziel dieses Projekts ist es, einen gegebenen Text zu analysieren und einer passenden Kategorie zuzuordnen, um das entsprechende Themengebiet zu identifizieren.

## Vorgehensweise
Zunächst wurde die Aufgabenstellung sorgfältig analysiert, um sicherzustellen, dass alle Anforderungen klar verstanden wurden. Dies ermöglichte es, einen strukturierten Ansatz zu entwickeln und den Fokus während der Umsetzung beizubehalten.

Im nächsten Schritt erfolgte eine umfassende Einarbeitung in die Thematik. Dazu wurde eine detaillierte Recherche durchgeführt, die folgende Quellen umfasste:

- Wissenschaftliche Arbeiten
- Fachliteratur zu maschinellem Lernen und Natural Language Processing (NLP)
- Anleitungen und Dokumentationen zu Frameworks wie TensorFlow, Scikit-learn und PyTorch
- Blogbeiträge und Online-Artikel
- Lehrvideos und Tutorials
- Darüber hinaus wurden bestehende Ansätze zur Textklassifikation untersucht, um bewährte Methoden und Strategien zu identifizieren.

## Ergebnisse
Die entwickelten Modelle konnten Texte erfolgreich erkennen und den entsprechenden Themengebieten zuordnen. Allerdings traten einige Herausforderungen auf:

- Themenbereiche, die in den Trainingsdaten seltener vorkamen, wurden nicht zuverlässig vorhergesagt.
- Bei längeren Texten kam es vereinzelt zu Fehlklassifikationen. Diese waren jedoch nicht vollkommen unzutreffend – beispielsweise wurden die Kategorien "Money" und "Earn" gelegentlich verwechselt, da sie thematisch eng miteinander verbunden sind.
Diese Erkenntnisse liefern wertvolle Ansätze für mögliche Optimierungen in zukünftigen Iterationen des Modells.

## **Verbesserungspotenzial**  

Während der Bearbeitung wurde bewusst darauf verzichtet, Deep-Learning-Algorithmen einzusetzen, da diese sehr rechenintensiv sind. Zudem werden Deep-Learning-Modelle häufig in Cloud-Umgebungen genutzt, was insbesondere bei Skalierung zu hohen Kosten führen kann.  

Stattdessen fiel die Entscheidung auf den **Random-Forest-Algorithmus**, um zu überprüfen, ob mit einem weniger rechenintensiven Modell vergleichbare oder sogar bessere Ergebnisse erzielt werden können.  

Da die Bearbeitungszeit auf drei Stunden begrenzt war, konnte keine umfassende Datenanalyse durchgeführt werden. Mit mehr Zeit wäre eine detailliertere Untersuchung des Datensatzes möglich gewesen. Ein zentrales Problem war, dass viele Artikel keine zugewiesenen **"Topics"** hatten. Beim **Data Preprocessing** wurden daher alle Zeilen mit fehlenden Werten entfernt, was die Datenverteilung möglicherweise verzerrt hat.  

Darüber hinaus blieben zahlreiche **Metadaten** ungenutzt, die potenziell zur Verbesserung der Klassifikation hätten beitragen können, wie beispielsweise:  
- **Zeitstempel der Veröffentlichung** – Mögliche saisonale Trends oder thematische Häufungen in bestimmten Zeiträumen.  
- **Autor** – Bestimmte Autoren könnten auf spezifische Themen spezialisiert sein, was als zusätzliches Merkmal genutzt werden könnte.  

Eine kurze Datenanalyse hat gezeigt, dass die Daten **stark unausgewogen** sind. Themen wie *acq*, *earn*, *money*, *grain* und *wheat* sind stark vertreten, während Kategorien wie *meal-feed*, *lead* oder *fuel* weniger als zehnmal im Datensatz vorkommen. Diese **Datenungleichgewicht** kann zu verzerrten Vorhersagen führen, da das Modell dazu neigt, Mehrheitsklassen zu bevorzugen.  

### **Mögliche Verbesserungen**  
Im weiteren Verlauf der Bearbeitung könnte der Einsatz von **Deep-Learning-Algorithmen** in Betracht gezogen werden, sofern mehr Zeit zur Verfügung stünde.  
Vielversprechende Ansätze wären:  
- **BERT** (Bidirectional Encoder Representations from Transformers): Ein leistungsstarkes Modell für die Verarbeitung natürlicher Sprache, das kontextbezogene Wortrepräsentationen nutzt und somit besonders geeignet für die semantische Analyse von Texten ist.  
- **LSTM** (Long Short-Term Memory): Ein rekurrentes neuronales Netz (RNN), das speziell für die Verarbeitung längerer Textsequenzen entwickelt wurde. 

Eine detaillierte Evaluierung könnte zeigen, inwiefern der Einsatz von **BERT** oder **LSTM** die Klassifikationsgenauigkeit im Vergleich zu klassischen Machine-Learning-Ansätzen verbessert.  

### **Code-Optimierung**  
Eine weitere Optimierung wäre es, den bestehenden Code in eine **Klassenstruktur** zu überführen, anstatt einzelne Funktionen zu nutzen. Dadurch könnte die Modularität und Wiederverwendbarkeit des Codes verbessert werden.