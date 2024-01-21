import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from top2vec import Top2Vec


"""
    a)  Programmieren Sie eine Python-Funktion für das Ranking von Art Stories im 
        Word2Vec Embedding, die einer Anfrage durch einen Besucher Ihrer Kunst-Website 
        am ähnlichsten sind.
    b)  Beurteilen Sie den Einfluss der Erweiterung auf das Doc2Vec 
        Embedding auf die Ergebnisqualität.
    c)  Programmieren Sie einen Empfehlungsdienst für Art Stories durch Clustering mit Top2Vec.
"""

allWordsRanking = []
vectors = [] # List where all the bag-of-words will be kept in
titles = []


def createBagOfWords(wordList):
    """ This method will receive a list of words, which are contained in a document.
        After that it will create a dictionary, where every word, is the key, and the value is the count of appearances in the document. 
        The new bag-of-words vector will be added to the global vectors list.
    """

    newVec = {}
    for word in allWordsRanking:
        if word in wordList:
            newVec[word] = wordList.count(word)
        else:
            newVec[word] = 0
    vectors.append(newVec) 

def text_to_word_list(text_to_clean):
    """ This method will receive a string, and returns the same string as a list of words, cleaned of any syntactical symbols like commas, parentheses, semicoli, ...
    """
    # first remove line breaks in the text
    string_without_linebreaks = text_to_clean.replace('\\n', ' ')
    # then remove NON alphabetical and NON numerical values
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', ' ', string_without_linebreaks)
    # replace double spaces and then turn to lowercase
    cleaned_string = cleaned_string.replace('  ', ' ').lower()
    # split the string into a list of words
    listOfWords = cleaned_string.split(" ")
    return listOfWords

def collectAllWords(wordsList):
    """ This method, will go through every word in the given documents, and put those words into a list of words.
    """
    for word in wordsList:
        if word not in allWordsRanking:
            allWordsRanking.append(word)

def read_document_text(path = "./Aufgabe 1/art_stories_examples.csv"):
    """ This method reads the data of a file which is a CSV, and has the content Semikolon(;) separated. 
        The columns are:  ContentType - Title - ShortDescriptionText - Text
    """
    # read file into the dataframe
    dataframe = pd.read_csv(path, sep=";")

    for title in dataframe['Title'].tolist():
        titles.append(title)

    # apply clean method on every cell
    # save every word to allWords list
    # create bag-of-words vector for each dataframe row / each document
    for column in ["Title", "ShortDescriptionText", "Text"]:
        dataframe[column] = dataframe[column].apply(text_to_word_list)
        collectAllWords(dataframe[column][0])

    for index, series in dataframe.iterrows():
        base = series["Title"]
        base.extend(series["ShortDescriptionText"])
        base.extend(series["Text"])
        createBagOfWords(base)

# a)
def rank_art_stories_python_function(user_query_string):
    # Prepare the user query
    user_query_words = text_to_word_list(user_query_string)

    user_query_vector = {}
    for word in allWordsRanking:
        user_query_vector[word] = user_query_words.count(word)

    dataframe_blueprint = {
        "Query" : [],
        "Title" : [],
        "Cosine Similarity": [],
        "Euclidian Distance": []
    }

    dataframe = pd.DataFrame(dataframe_blueprint)
    for vector in vectors:
        similarity = cosine_similarity([list(user_query_vector.values())], [list(vector.values())])[0][0]
        distance = euclidean_distances([list(user_query_vector.values())], [list(vector.values())])[0][0]
        dict = {"Query": user_query_string, "Title": titles[vectors.index(vector)], "Cosine Similarity": similarity, "Euclidian Distance": distance}
        dataframe = dataframe.append(dict, ignore_index = True)
        
    return dataframe.sort_values(by="Euclidian Distance", ascending=True)

# c)
def recommend_art_stories_python_function(identifier_of_visited_art_stories_list):
    """Based of the created clusters of the top2vec we search in 
    which cluster the visited art stories where assigned and therefore 
    recommend the top 3 similar documents (top 1 of each cluster)"""
    
    # Read the dataframe
    dataframe = pd.read_csv('./Aufgabe 1/art_stories_examples.csv', delimiter=';')

    # create the document for the Top2Vec clustering
    documents = dataframe['Text'].tolist()
    documents = documents * 10  # of course nonsense later expand the art stories csv

    # create the top2vec model based on the art story document (art_stories_example.csv)
    top2vec_model = Top2Vec(documents, embedding_model='universal-sentence-encoder', min_count=1, speed='fast-learn', workers=4)

    # Get the clusters where the visited art stories where assigned to.
    visited_vector_topics = top2vec_model.get_documents_topics(identifier_of_visited_art_stories_list)

    # Get the cluster indices
    visited_topic_indices = visited_vector_topics[0]
    # Now for each cluster get me the most similar document based on the cluster
    recommendation = []
    for visited_topic_index in visited_topic_indices:
        # Show top 1 document for each topic
        similar_documents_based_on_topic = top2vec_model.search_documents_by_topic(visited_topic_index, num_docs=1)
        recommendation.append(similar_documents_based_on_topic[0])
    return recommendation

visited_art_stories = [1, 4, 7]
recommendations = recommend_art_stories_python_function(visited_art_stories)
print("Recommended art stories:")
for recommendation in recommendations:
    print(recommendation)
