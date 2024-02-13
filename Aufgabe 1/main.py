import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from top2vec import Top2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


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

def clean_text(text_to_clean):
    """ This method will receive a string, and returns the same string as a list of words, cleaned of any syntactical symbols like commas, parentheses, semicoli, ...
    """
    # first remove line breaks in the text
    string_without_linebreaks = text_to_clean.replace('\\n', ' ')
    # then remove NON alphabetical and NON numerical values
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', ' ', string_without_linebreaks)
    # replace double spaces and then turn to lowercase
    cleaned_string = cleaned_string.replace('  ', ' ').lower()
    return cleaned_string

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
    # turn user query into a list of words
    user_query_words = text_to_word_list(user_query_string)

    # generate the bag of words vector for the user query it self
    user_query_vector = {}
    for word in allWordsRanking:
        user_query_vector[word] = user_query_words.count(word)

    # the structure of the dataframe which will be returned
    dataframe_blueprint = {
        # shows the query that was looked for
        "Query" : [],
        # shows the title of the document
        "Title" : [],
        # the value of cosine similarity. Cosine similarity is the angular difference of two vectors. Here the two vectors A, B are put into this equation: (A*B) / Length of A * Lenght of B. 
        # Problem with cosine similarity is, that it really only compares with the angles. It does NOT include the length of a vector. 
        # so lets say we have one vector A1. And then A2 and A3. A2 has the length of 3 and A3 has the length of 5000. The cosine similarity value is the same if compared to A1
        "Cosine Similarity": [],
        # the value of euclidean distance between the document vector and the query vector. It represents the distance of two points.
        # The euclidean distance is calculated like this: ((x[0] - y[0])**2 + (x[1] - y[1])**2 + ... + (x[n] - y[n])**2)**0.5
        "Euclidian Distance": [],
        # To generate a representation of similarity, we use the cosine similarity and euclidian distance to calculate a new value.
        # so we combine the angular difference with the distance difference.
        # we combine them like this: cosine similairty + 1/(euclidean distance + alpha)
        # we use the inverted euclidean distance, because the euclidean distance would be not good to multiply with cosine similarity
        # high euclidean distances should correspond to lower similarity. When we take the inverted distance, we have higher values to indicate higher similarity
        "Similarity Value": [],
    }

    dataframe = pd.DataFrame(dataframe_blueprint)

    # alpha is a value which is used for the inverted euclidean distance, to avoid division by zero.
    # a very low number is used to not change the value by too much
    alpha = 1e-10 # 1 * 10^-10 = 0.0000000001

    # iterate through every vector of each document
    for vector in vectors:
        # generate cosine similarity between the user query vector and the document vector
        similarity = cosine_similarity([list(user_query_vector.values())], [list(vector.values())])[0][0]
        # calculate the euclidean distance between the user query vector and the document vector
        distance = euclidean_distances([list(user_query_vector.values())], [list(vector.values())])[0][0]
        # create a dictionary, which represents one row of the dataframe
        dict = {"Query": user_query_string, "Title": titles[vectors.index(vector)], "Cosine Similarity": similarity, "Euclidian Distance": distance, "Similarity Value": similarity + 1 / (distance + alpha)}
        # append the dictionary into the dataframe
        dataframe = dataframe.append(dict, ignore_index = True)
        
    # return the dataframe, after sorting it. The first row will be the most similar document
    return dataframe.sort_values(by="Similarity Value", ascending=False)

"""
    Prüfungsaufgabe 1
    Aufgabe b)

    Beurteilen Sie den Einfluss der Erweiterung auf das Doc2Vec EMbedding auf die Ergebnisqualität.

    Zuvor beim Word2Vec Embedding, werden nur die Wörter in den Dokumenten gezählt. Je ähnlicher die Wort-Vektoren sind, desto ähnlicher die Dokumente.

    Beim Doc2Vec, geht es jedoch nicht nur um die Wort-Frequenzen sondern auch um Struktur. Es werden also Paragraphen Teilweise als Vektoren repräsentiert.
    Der Vorteil hier ist es, dass man hiermit semantische Ähnlichkeiten näher untersuchen kann. Bei Doc2Vec werden also Word2Vec Logik genutzt und auch eine Repräsentation des gesamten Dokuments. 

    Die Ergebnisqualität würde sich erhöhen. Bei Doc2Vec arbeitet man mit Semantik. ALso muss man die Tokenisierten Dokumente, Taggen. 
    Und mithilfe dieser Tags, kann man den Kontext der Dokumente mit berücksichtigen. Mithilfe dieser Tags, wird das Modell trainiert, und das
    Modell kann eindeutige Zusammenhänge zwischen Kontext und Dokument erfassen.

    Die Ergebnisqualität würde also steigen, je nachdem wie viele Dokumente man hat. Um das Modell zu trainieren sollte man je nach Modell, mehr Dokumente haben.
    Bei dem Modell von gensim sollten z.B. 15 Dokumente von diesem Volumen wie hier ausreichen.
"""

def doc2Vec_model(user_query_string):

    # Read the given documents into array. Here the bag of words from earlier won't be used, and we will make sure to use the convenience of provided methods.
    documents = []
    with open('./Aufgabe 1/art_stories_examples.csv', 'r', encoding="utf-8") as file:
        for line in file.readlines()[1:]:
            lineSecs = line.split(";")
            documents.append(clean_text(lineSecs[1]) + clean_text(lineSecs[2]) + clean_text(lineSecs[3]))


    # Tokenize and tag the documents
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)]

    # Create instance of Doc2Vec and provide tagged data to train it
    model = Doc2Vec(vector_size=20, min_count=1, epochs=100)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Create Vector for user query, with infer_vector method, because the words in the user query can not assure if they are in the vocabulary
    user_vector = model.infer_vector(word_tokenize(user_query_string.lower()))

    # Go through every document and generate Ranking. The ranking will be a list of tuples, where the first value is the document and the second value is the rank.
    ranks = []
    for i in range(len(documents)):
        doc_vector = model.docvecs[str(i)]
        rank = cosine_similarity([user_vector.tolist()], [doc_vector.tolist()])[0][0]
        ranks.append((documents[i], rank))

    # Sort by rank
    ranks.sort(key=lambda x: x[1])

    return ranks


# c)
def recommend_art_stories_python_function(identifier_of_visited_art_stories_list):
    """Based of the created clusters of the top2vec we search in 
    which cluster the visited art stories where assigned and therefore 
    recommend the top 3 similar documents (top 1 of each cluster)"""
    
    # Read the documents into a dataframe with pandas
    dataframe = pd.read_csv('./Aufgabe 1/art_stories_examples.csv', delimiter=';')

    # create the document for the Top2Vec clustering
    documents = dataframe['Text'].tolist()
    # in this case we multiply the amount of documents by 10. The library we use needs enough documents to train the model with
    # this changes the value in this case, but because we didnt have a big enough dataset, this is what we used.
    # when testing and having a big enough data set please comment the line
    documents = documents * 10 

    # Here we give Top2Vec our documents, and it will create a model out of every document.
    # We use the universal-sentence-encoder, because those are pretrained and it will run faster.
    # there is another model 'doc2vec' but that would take longer to generate
    top2vec_model = Top2Vec(documents, embedding_model='universal-sentence-encoder', min_count=1, speed='fast-learn', workers=4)

    # after the model is trained, we will get the topic for each document. 
    # this method wont only return the topic. it will return a list of items in this order:
    #  a list of numbers which represent the topic numbers of the document
    #  a list of scores, which represent the similarity of the document to the topics
    #  for each topic the top 50 words
    #  a list of word scores for each topic and document, which is represented by cosine similarity
    visited_vector_topics = top2vec_model.get_documents_topics(identifier_of_visited_art_stories_list)

    # get the list of topic indices
    visited_topic_indices = visited_vector_topics[0]

    recommendation = []
    # go through every topic
    for visited_topic_index in visited_topic_indices:
        # get the most similar document in the topic
        similar_documents_based_on_topic = top2vec_model.search_documents_by_topic(visited_topic_index, num_docs=1)
        # add the document, which is the most similar in the topic to the recommendations
        recommendation.append(similar_documents_based_on_topic[0])
    return recommendation

"""
visited_art_stories = [1, 4, 7]
recommendations = recommend_art_stories_python_function(visited_art_stories)
print("Recommended art stories:")
for recommendation in recommendations:
    print(recommendation)
"""