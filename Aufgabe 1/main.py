"""
    a)  Programmieren Sie eine Python-Funktion für das Ranking von Art Stories im 
        Word2Vec Embedding, die einer Anfrage durch einen Besucher Ihrer Kunst-Website 
        am ähnlichsten sind.
    b)  Beurteilen Sie den Einfluss der Erweiterung auf das Doc2Vec 
        Embedding auf die Ergebnisqualität.
    c)  Programmieren Sie einen Empfehlungsdienst für Art Stories durch Clustering mit Top2Vec.
"""

corpus = []

def read_document_text(path):
    document_lines = []
    with open(path, "r") as document:
        for line in document:
            document_lines.append(line)
            words = line.split(" ")
            for word in words:
                if word not in corpus:
                    corpus.append(word)
    document.close()


def rank_art_stories_python_function(user_query_string):
    pass

def recommend_art_stories_python_function(identifier_of_visited_art_stories_list):
    pass