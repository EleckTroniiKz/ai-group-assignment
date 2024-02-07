import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import json

# The flat interaction data is loaded and the vectores with the words were prepaired only for the person with the
# matching id
R_data = pd.read_csv("User-Item_Interaction.csv", delimiter=';')
S_data = dict()
for user in R_data["USERNAME"].unique():
    user = int(user)
    data = R_data.loc[R_data["USERNAME"] == user]
    beschreibungen_dict = pd.read_csv("Item-Description.csv", delimiter=';')
    beschreibungen_dict = beschreibungen_dict.set_index("ARTWORK_ID")
    beschreibungen_dict = beschreibungen_dict.to_dict(orient='dict')["BESCHREIBUNG"]

    # TfidfVectorizer initialisieren und Stop-Wörter übergeben
    tfidf_vectorizer = TfidfVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))

    # Beschreibungen der Kunstwerke in eine Liste umwandeln
    beschreibungen = [beschreibungen_dict[artwork_id] for artwork_id in data['ARTWORK_ID']]

    # TF-IDF-Matrix der Beschreibungen erstellen
    tfidf_matrix = tfidf_vectorizer.fit_transform(beschreibungen)

    # Betrachtungsdauer der Kunstwerke durch die Kunden
    betrachtungsdauer = data['TIMEDIFF_SECS']

    # Konvertieren Sie betrachtungsdauer in ein Numpy-Array und fügen Sie eine neue Achse hinzu
    betrachtungsdauer_np = betrachtungsdauer.to_numpy().reshape(-1, 1)

    # Gewichtung der TF-IDF-Matrix basierend auf der Betrachtungsdauer
    gewichtete_tfidf_matrix = tfidf_matrix.multiply(betrachtungsdauer_np)

    # Kundenprofile erstellen
    kundenprofile = gewichtete_tfidf_matrix.sum(axis=0)

    # Normalisierung der Kundenprofile
    kundenprofile_normalized = kundenprofile / kundenprofile.sum()

    # Wörter aus der TF-IDF-Matrix erhalten
    woerter = tfidf_vectorizer.get_feature_names_out()

    # Gewichtungen aus der gewichteten TF-IDF-Matrix extrahieren
    gewichtungen = kundenprofile_normalized.A.ravel()

    # Wörter und zugehörige Gewichtungen ausgeben, wobei Füllwörter ignoriert werden
    sorted_words = sorted(zip(woerter, gewichtungen), key=lambda x: x[1], reverse=True)

    # Die ersten 25 Elemente auswählen
    top_words = sorted_words[:10]
    S_data[user] = top_words
with open("user_profile.txt", "w") as profile:
    json.dump(S_data, profile)
