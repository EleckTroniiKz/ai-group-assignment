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

def clean_text(text_to_clean):
    """ This method will receive a string, and returns the same string, cleaned of any syntactical symbols like commas, parentheses, semicoli, ...
    """
    # first remove line breaks in the text
    string_without_linebreaks = text_to_clean.replace('\n', ' ')
    # then remove NON alphabetical and NON numerical values
    cleaned_string = re.sub(r'[a-zA-Z0-9]', ' ', string_without_linebreaks)
    return cleaned_string

def read_document_text(path="./art_stories_examples.csv"):
    """ This method reads the data of a file which is a CSV, and has the content Semikolon(;) separated. 
        The columns are:  ContentType - Title - ShortDescriptionText - Text
    """
    # CAN MACHT DIESE METHODE GRAD BIDDE NIT ANFASSEN :D
    # read file into the dataframe
    dataframe = pd.read_csv(path, sep=";")
    for row in dataframe.iterrows():
        print(row)

    pass

read_document_text()

def rank_art_stories_python_function(user_query_string):
    pass

def recommend_art_stories_python_function(identifier_of_visited_art_stories_list):
    pass