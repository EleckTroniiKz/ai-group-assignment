import pandas as pd
import re

"""
    a)  Programmieren Sie eine Python-Funktion für das Ranking von Art Stories im 
        Word2Vec Embedding, die einer Anfrage durch einen Besucher Ihrer Kunst-Website 
        am ähnlichsten sind.
    b)  Beurteilen Sie den Einfluss der Erweiterung auf das Doc2Vec 
        Embedding auf die Ergebnisqualität.
    c)  Programmieren Sie einen Empfehlungsdienst für Art Stories durch Clustering mit Top2Vec.
"""

corpus = []

def create_bag_of_words(rowText):
    pass

def text_to_word_list(text_to_clean):
    """ This method will receive a string, and returns the same string as a list of words, cleaned of any syntactical symbols like commas, parentheses, semicoli, ...
    """

    # first remove line breaks in the text
    string_without_linebreaks = text_to_clean.replace('\n', '')
    # then remove NON alphabetical and NON numerical values
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', ' ', string_without_linebreaks)
    # replace double spaces and then turn to lowercase
    cleaned_string = cleaned_string.replace('  ', ' ').lower()
    return cleaned_string.split(" ")

def read_document_text(path="./art_stories_examples.csv"):
    """ This method reads the data of a file which is a CSV, and has the content Semikolon(;) separated. 
        The columns are:  ContentType - Title - ShortDescriptionText - Text
    """
    # read file into the dataframe
    dataframe = pd.read_csv(path, sep=";")

    # apply clean method on every cell
    for column in ["ShortDescriptionText", "Text"]:
        dataframe[column] = dataframe[column].apply(text_to_word_list)
    
    return dataframe

def rank_art_stories_python_function(user_query_string):
    pass

def recommend_art_stories_python_function(identifier_of_visited_art_stories_list):
    pass