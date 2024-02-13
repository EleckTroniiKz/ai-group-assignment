import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import json

"""
This script generates the user profiles. It generates the words with a score. 
The score is relative to the total time watched.
For example, "wooden 0.2" means that the user spent 20% of his time looking at wooden artworks.

We decided to make the whole thing content-based because we have more data records that describe a single user and 
their behaviour than data records in which several users interact.  Alternatively, connections to other users could 
also be made via the jointly viewed images. As an approach for this, the similarity of the vectors of the individual 
users could be calculated. The current data contains too few users for this.
"""

# The flat interaction data is loaded and the vectors with the words were prepared only for the person with the
# matching id
R_data = pd.read_csv("User-Item_Interaction.csv", delimiter=';')
S_data = dict()
for user in R_data["USERNAME"].unique():
    user = int(user)
    data = R_data.loc[R_data["USERNAME"] == user]
    beschreibungen_dict = pd.read_csv("Item-Description.csv", delimiter=';')
    beschreibungen_dict = beschreibungen_dict.set_index("ARTWORK_ID")
    beschreibungen_dict = beschreibungen_dict.to_dict(orient='dict')["BESCHREIBUNG"]

    # the input of the follwoing section is inspired by the documentation
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html and medium
    # https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a

    # a list with common english words is loaded. These were not included in the ranking.
    tfidf_vectorizer = TfidfVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
    description = [beschreibungen_dict[artwork_id] for artwork_id in data['ARTWORK_ID']]
    # Generates the document-term matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(description)
    # In this matrix the seconds watched were multiplied.
    betrachtungsdauer_np = data['TIMEDIFF_SECS'].to_numpy().reshape(-1, 1)
    gewichtete_tfidf_matrix = tfidf_matrix.multiply(betrachtungsdauer_np)

    # the customer profile is created and converted to relative numbers
    kundenprofile = gewichtete_tfidf_matrix.sum(axis=0)
    kundenprofile_normalized = kundenprofile / kundenprofile.sum()

    # the words and the score were extracted and sorted
    woerter = tfidf_vectorizer.get_feature_names_out()
    gewichtungen = kundenprofile_normalized.A.ravel()
    sorted_words = sorted(zip(woerter, gewichtungen), key=lambda x: x[1], reverse=True)

    # the top 10 ranked words per user were stored
    top_words = sorted_words[:10]
    S_data[user] = top_words
with open("user_profile.txt", "w") as profile:
    json.dump(S_data, profile)
