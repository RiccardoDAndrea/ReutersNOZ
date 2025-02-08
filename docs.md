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
Zu lange texte wiesen drauf hin das die Klassifikation nicht zutreffen waren da keine Vorerhsage getroffen wurden

## **Verbesserungspotenzial**  

Während der Bearbeitung wurde bewusst darauf verzichtet, Deep-Learning-Algorithmen zu verwenden, da diese sehr rechenintensiv sind. Zudem werden Deep-Learning-Modelle häufig in Cloud-Umgebungen eingesetzt, was insbesondere bei Skalierung zu hohen Kosten führen kann.  

Stattdessen fiel die Entscheidung auf den **Random Forest Algorithmus**, um zu überprüfen, ob mit einem weniger rechenintensiven Modell vergleichbare oder sogar bessere Ergebnisse erzielt werden können.  

Da die Bearbeitungszeit auf drei Stunden begrenzt war, konnte keine umfassende Datenanalyse durchgeführt werden. Mit mehr Zeit wäre eine detailliertere Untersuchung des Datensatzes möglich gewesen. Ein zentrales Problem war, dass viele Artikel keine zugewiesenen **"Topics"** hatten. Beim **Data Preprocessing** wurden daher alle Zeilen mit fehlenden Werten entfernt, was die Datenverteilung möglicherweise verzerrt hat.  

Darüber hinaus blieben zahlreiche **Metadaten** ungenutzt, die potenziell zur Verbesserung der Klassifikation hätten beitragen können, wie beispielsweise:  
- **Zeitstempel der Veröffentlichung** – Mögliche saisonale Trends oder thematische Häufungen in bestimmten Zeiträumen.  
- **Autor** – Bestimmte Autoren könnten auf spezifische Themen spezialisiert sein, was als zusätzliches Merkmal genutzt werden könnte.  

Eine kurze Datenanalyse hat gezeigt, dass die Daten **sehr unausgewogen** sind. Themen wie *acq*, *earn*, *money*, *grain* und *wheat* sind stark vertreten, während Kategorien wie *meal-feed*, *lead* oder *fuel* weniger als zehnmal im Datensatz vorkommen. Diese **Datenungleichgewicht** kann zu verzerrten Vorhersagen führen, da das Modell dazu neigt, Mehrheitsklassen zu bevorzugen.  

Im weiteren Verlauf der Bearbeitung könnte ein **Deep-Learning-Algorithmus** getestet werden, sofern mehr Zeit zur Verfügung stünde.  
Mögliche Ansätze wären der Einsatz von **BERT** oder **LSTM** für die Textklassifikation, da sowohl wissenschaftliche Arbeiten als auch Blogeinträge deren hohe Zuverlässigkeit in ähnlichen Anwendungsfällen hervorheben.  

Während **BERT** kontextbezogene Wortrepräsentationen nutzt und somit besonders leistungsfähig bei der Erkennung semantischer Zusammenhänge ist, bietet sich **LSTM (Long Short-Term Memory)** speziell für die Verarbeitung längerer Textsequenzen an. LSTMs können sich über längere Textabschnitte hinweg Informationen merken und sind daher für die Analyse von zusammenhängenden Texten geeignet.  

Eine detaillierte Evaluierung könnte zeigen, inwiefern **BERT** oder **LSTM** im Vergleich zu klassischen Machine-Learning-Ansätzen die Klassifikationsgenauigkeit verbessern.  

