# Imports

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import seaborn as sns
import rdflib
import wikipediaapi
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import spacy
from spacy.tokens.doc import Doc 
from spacy.pipeline.sentencizer import Sentencizer
from spacy.language import Language
from spacy import displacy
import textacy
import networkx as nx
import plotly.graph_objs as go
import dateparser
import PyPDF2
from mpl_toolkits.mplot3d import Axes3D

"""
    The provided code was very hardly readable, because of the lack of comments and the lack of structure.
    Because it was a JupyterNotebook before, it was not as efficient to work with in a "normal" Development environment.
    For that reason, We decided to restructure the code and add comments to make it more readable and understandable.
    Also we added some print statements to make it more clear, what the code is doing at the moment.
    The code is now structured in a way, that it is easier to understand and to work with.
    Especially with extracting specific parts to methods, to create more structure.
"""

class ArtKnowledgeGraph:

    def __init__(self, path, isPDF = False, topic="Art Basel Fair") -> None:
        dataSource = None
        text = ""
        if(isPDF):
            pdfFileObj = open(path, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            for page in pdfReader.pages:
                text += page.extract_text()
            pdfFileObj.close()
        else:
            textDataFrame = pd.read_csv(path, sep=",")
            textDataFrame['STORY_TEXT'] = textDataFrame['STORY_TEXT'].apply(self.clean_text_from_html_content)
            for id, line in textDataFrame.iterrows():
                text += line[1]
        
        print("Initialized!")

        self.text = text
        self.doc = []
        self.nlp = spacy.load("en_core_web_sm")
        self.lst_docs = []
        self.lst_entities = []
        self.lst_relations = []
        self.topic = topic

    # This method will take the generated graph and save it as a .ttl file
    # For the conveniance of the devs (us :) ), a library called RDFLib is used
    # the graph is generated using the networkx library
    def generate_ttl_file(self, GRAPH, output_file="output.ttl"):
        graph = rdflib.Graph()

        for edge in GRAPH.edges(data=True):
            subj = rdflib.URIRef(edge[0])
            pred = rdflib.URIRef(edge[2]['relation'])
            obj = rdflib.URIRef(edge[1])
            graph.add((subj, pred, obj))
        graph.serialize(destination=output_file, format="turtle")

    # This method will take the text and clean it from any html content
    # It will then save the cleaned text to a file called "Output.txt"
    # The gile is not needed, but during development it was helpful to see, the result of the cleanse
    def clean_text_from_html_content(self, string):
        text = re.sub('<.*?>', ' ', string)
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text)
        re.sub(r"[^\w\s\.\!\?\(\)]", "", text)

        with open("C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Aufgabe 3\\ArtKnowledgeGraph\\ArtKnowledgeGraph\\Output.txt", "w", encoding="utf-8") as t_file:
            t_file.write(text)
            t_file.close()
        return text

    def word_freq(self, txt, ngrams=[1,2,3], top=10, figsize=(10,7)):
        print("Calculating word frequency.")
        lst_tokens = nltk.tokenize.word_tokenize(txt)
        ngrams = [ngrams] if type(ngrams) is int else ngrams
        
        ## calculate
        dtf_freq = pd.DataFrame()
        for n in ngrams:
            dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
            dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
            dtf_n["ngrams"] = n
            if dtf_freq.empty:
                dtf_freq = dtf_n
            else:
                dtf_freq = dtf_freq.append(dtf_n)
        dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x) )
        dtf_freq = dtf_freq.sort_values(["ngrams","freq"], ascending=[True,False])
        
        ## plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                    data=dtf_freq.groupby('ngrams')[["ngrams","freq","word"]].head(top))
        ax.set(xlabel=None, ylabel=None, title="Most frequent words")
        ax.grid(axis="x")
        plt.show()
        return dtf_freq

    def process_text(self, chunks=1000000):
        print("Process text. This might take some time.")
        for i in range(0, len(self.text), chunks):
            self.doc.append(self.nlp(self.text[i: i+ chunks]))
        self.lst_docs = [sent for doc in self.doc for sent in doc.sents]
        print("Total sentences: " + str(len(self.lst_docs)))

    def extract_entities(self, doc):
        
        a, b, prev_dep, prev_txt, prefix, modifier = "", "", "", "", "", ""
        for token in doc:
            if token.dep_ != "punct":
                ## prexif --> prev_compound + compound 
                if token.dep_ == "compound":
                    prefix = prev_txt +" "+ token.text if prev_dep == "compound" else token.text
                
                ## modifier --> prev_compound + %mod 
                if token.dep_.endswith("mod") == True:
                    modifier = prev_txt +" "+ token.text if prev_dep == "compound" else token.text
                
                ## subject --> modifier + prefix + %subj 
                if token.dep_.find("subj") == True:
                    a = modifier +" "+ prefix + " "+ token.text
                    prefix, modifier, prev_dep, prev_txt = "", "", "", ""
                
                ## if object --> modifier + prefix + %obj 
                if token.dep_.find("obj") == True:
                    b = modifier +" "+ prefix +" "+ token.text
                
                prev_dep, prev_txt = token.dep_, token.text
        
        # clean
        a = " ".join([i for i in a.split()])
        b = " ".join([i for i in b.split()])
        return (a.strip(), b.strip())

    def extract_relation(self, doc):
        matcher = spacy.matcher.Matcher(self.nlp.vocab)
        p1 = [{'DEP':'ROOT'}, 
            {'DEP':'prep', 'OP':"?"},
            {'DEP':'agent', 'OP':"?"},
            {'POS':'ADJ', 'OP':"?"}] 
        matcher.add(key="matching_1", patterns=[p1]) 
        matches = matcher(doc)
        k = len(matches) - 1
        span = doc[matches[k][1]:matches[k][2]] 
        return span.text
    
    def utils_parsetime(self, to_parse_text):
        x = re.match(r'.*([1-3][0-9]{3})', to_parse_text) #<-- check if there is a year
        if x is not None:
            try:
                dt = dateparser.parse(to_parse_text)
            except:
                dt = np.nan
        else:
            dt = np.nan
        return dt

    def process(self, word):
        print("Starting process")
        show_word_frequency = True


        self.process_text()

        if(show_word_frequency):
            self.word_freq(self.text, ngrams=[3], top=20, figsize=(10, 7))
        
        for token in self.lst_docs[3]:
            print(token.text, "-->", "pos: "+token.pos_, "|", "dep: "+token.dep_, "")
        
        for tag in self.lst_docs[3].ents:
            print(tag.text, f"({tag.label_})")
            
        print("Extracting entitites")
        self.lst_entities = [self.extract_entities(i) for i in self.lst_docs]
        print("Extracting relations")
        self.lst_relations = [self.extract_relation(i) for i in self.lst_docs]

        print("Collecting Attribute List Content")
        lst_attributes = []
        for x in self.lst_docs:
            attribute = ""
            for tag in x.ents:
                attribute += tag.text if tag.label_=="DATE" else ""
            lst_attributes.append(attribute)
        
        dataframe_base = pd.DataFrame({"text":[doc.text for doc in self.lst_docs], "entity":[i[0] for i in self.lst_entities], "relation": self.lst_relations, "object":[i[1] for i in self.lst_entities], "attribute": lst_attributes })
        dictionary_tmp = {"id":[], "text":[], "entity":[], "relation":[], "object":[]}

        print("Filling Dictionary with Id, Text, Entity, Object, Relation")
        for n, sentence in enumerate(self.lst_docs):
            lst_generators = list(textacy.extract.subject_verb_object_triples(sentence))
            for sent in lst_generators:
                subj = "_".join(map(str, sent.subject))
                obj  = "_".join(map(str, sent.object))
                relation = "_".join(map(str, sent.verb))
                dictionary_tmp["id"].append(n)
                dictionary_tmp["text"].append(sentence.text)
                dictionary_tmp["entity"].append(subj)
                dictionary_tmp["object"].append(obj)
                dictionary_tmp["relation"].append(relation)
        
        dataframe_one = pd.DataFrame(dictionary_tmp)
        dataframe_one = dataframe_one[dataframe_one["object"].str.len() < 20]

        dictionary_tmp_one = {"id":[], "text":[], "date":[]}

        print("Filling another Dictionary with Id, Text and Date")
        for n,sentence in enumerate(self.lst_docs):
            lst = list(textacy.extract.entities(sentence, include_types={"DATE"}))
            if len(lst) > 0:
                for attr in lst:
                    dictionary_tmp_one["id"].append(n)
                    dictionary_tmp_one["text"].append(sentence.text)
                    dictionary_tmp_one["date"].append(str(attr))
            else:
                dictionary_tmp_one["id"].append(n)
                dictionary_tmp_one["text"].append(sentence.text)
                dictionary_tmp_one["date"].append(np.nan)
        
        dataframe_attributes = pd.DataFrame(dictionary_tmp_one)
        dataframe_attributes = dataframe_attributes[~dataframe_attributes["date"].isna()]

        """
        in the provided code here would be the same loop, but for Location. I skipped this, because in the provided code, the results of that loop were not used. So to lower the computation time, that part is skipped
        """

        GRAPH = nx.from_pandas_edgelist(dataframe_one.head(15), source="entity", target="object", edge_attr="relation", create_using=nx.DiGraph())

        print("Generating first ttl File")
        self.generate_ttl_file(GRAPH, "output_file.ttl")

        plt.figure(figsize=(15, 10))

        pos = nx.spring_layout(GRAPH, k=1)

        node_color, edge_color = "skyblue", "black"

        nx.draw(GRAPH, pos=pos, with_labels=True, node_color=node_color, edge_color=edge_color, cmap=plt.cm.Dark2, 
        node_size=2000, connectionstyle='arc3,rad=0.1')

        nx.draw_networkx_edge_labels(GRAPH, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(GRAPH,'relation'),
                                    font_size=12, font_color='black', alpha=0.6)
        #Plot graph of whole dataset:
        print("Showing first plot. Please close the View of the Plot, to continue the code execution.")
        plt.show()

        f = word
        tmp = dataframe_one[(dataframe_one["entity"] == f) | (dataframe_one["object"]==f)]
        GRAPH = nx.from_pandas_edgelist(tmp, source="entity", target="object", edge_attr="relation", create_using=nx.DiGraph())

        print("Generating new TTL File")
        self.generate_ttl_file(GRAPH, "output_2_file.ttl")

        plt.figure(figsize=(15,10))
        pos = nx.nx_agraph.graphviz_layout(GRAPH, prog="neato")

        node_color, edge_color = ["red" if node==f else "skyblue" for node in GRAPH.nodes], ["red" if edge[0]==f else "black" for edge in GRAPH.edges]

        nx.draw(GRAPH, pos=pos, with_labels=True, node_color=node_color, edge_color=edge_color, cmap=plt.cm.Dark2, node_size=2000, node_shape="o", connectionstyle='arc3,rad=0.1')

        nx.draw_networkx_edge_labels(GRAPH, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(GRAPH,'relation'), font_size=12, font_color='black', alpha=0.6)

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection="3d")
        pos = nx.spring_layout(GRAPH, k=2.5, dim=3)

        nodes = np.array([pos[v] for v in sorted(GRAPH) if v!=f])
        center_node = np.array([pos[v] for v in sorted(GRAPH) if v==f])

        edges = np.array([(pos[u],pos[v]) for u,v in GRAPH.edges() if v!=f])
        center_edges = np.array([(pos[u],pos[v]) for u,v in GRAPH.edges() if v==f])
        if(len(nodes.T) > 0): ax.scatter(*nodes.T, s=200, ec="w", c="skyblue", alpha=0.5)
        if(len(center_node.T) > 0): ax.scatter(*center_node.T, s=200, c="red", alpha=0.5)

        print("Linking Edges")
        for link in edges:
            ax.plot(*link.T, color="grey", lw=0.5)
        for link in center_edges:
            ax.plot(*link.T, color="red", lw=0.5)

        print("Adding Text")
        for v in sorted(GRAPH):
            ax.text(*pos[v].T, s=v)
        for u,v in GRAPH.edges():
            attr = nx.get_edge_attributes(GRAPH, "relation")[(u,v)]
            ax.text(*((pos[u]+pos[v])/2).T, s=attr)
        
        ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], yticklabels=[], zticklabels=[])
        ax.grid(False)

        print("Set axis tick locations")
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])

        print("Set the Nodes positions")
        pos = nx.spring_layout(GRAPH, k=0.2)
        for n,p in pos.items():
            GRAPH.nodes[n]['pos'] = p

        edge_x, edge_y = [], []
        arrows = []
        print("add Edges")
        for n,edge in enumerate(GRAPH.edges()):
            x0, y0 = GRAPH.nodes[edge[0]]['pos']
            x1, y1 = GRAPH.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            arrows.append([[x0,y0],[x1,y1]])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', showlegend=False)

        node_x, node_y = [], []
        print("add Nodes")
        for node in GRAPH.nodes():
            x, y = GRAPH.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', showlegend=False, marker={"showscale":False, "colorscale":'YlGnBu', "reversescale":True, "size":10, "line_width":2})

        link_text, node_text, node_color, node_size = [], [], [], []
        print("Add Relations")
        for adjacencies in GRAPH.adjacency():
            node_text.append(adjacencies[0])
            for dic in adjacencies[1].values():
                link_text.append(dic["relation"])

        node_trace.text = node_text
        edge_trace.text = link_text

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title=self.topic, showlegend=True, plot_bgcolor='white', hovermode='closest', width=800, height=800,xaxis={"visible":False}, yaxis={"visible":False})

        dataframe_attributes["dt"] = dataframe_attributes["date"].apply(lambda x: self.utils_parsetime(x))

        tmp = dataframe_one.copy()
        tmp["y"] = tmp["entity"]+" "+tmp["relation"]+" "+tmp["object"]

        dataframe_attributes = dataframe_attributes.merge(tmp[["id","y"]], how="left", on="id")
        dataframe_attributes = dataframe_attributes[~dataframe_attributes["y"].isna()].sort_values("dt", ascending=True).drop_duplicates("y", keep='first')
        dataframe_attributes.head()

        dates = dataframe_attributes["dt"].values
        names = dataframe_attributes["y"].values
        l = [10,-10, 8,-8, 6,-6, 4,-4, 2,-2]
        levels = np.tile(l, int(np.ceil(len(dates)/len(l))))[:len(dates)]

        fig, ax = plt.subplots(figsize=(20,10))
        ax.set(title=self.topic, yticks=[], yticklabels=[])

        ax.vlines(dates, ymin=0, ymax=levels, color="tab:red")
        ax.plot(dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w")

        for d,l,r in zip(dates,levels,names):
            ax.annotate(r, xy=(d,l), xytext=(-3, np.sign(l)*3), textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="bottom" if l>0 else "top")
        plt.xticks(rotation=90) 

        yyyy = "2023"
        dates = dataframe_attributes[dataframe_attributes["dt"]>yyyy]["dt"].values
        names = dataframe_attributes[dataframe_attributes["dt"]>yyyy]["y"].values
        l = [10,-10, 8,-8, 6,-6, 4,-4, 2,-2]
        levels = np.tile(l, int(np.ceil(len(dates)/len(l))))[:len(dates)]

        fig, ax = plt.subplots(figsize=(20,10))
        ax.set(title=self.topic, yticks=[], yticklabels=[])

        ax.vlines(dates, ymin=0, ymax=levels, color="tab:red")
        ax.plot(dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w")

        for d,l,r in zip(dates,levels,names):
            ax.annotate(r, xy=(d,l), xytext=(-3, np.sign(l)*3), textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="bottom" if l>0 else "top")

        plt.xticks(rotation=90) 
        #Plot filtered timeline
        plt.show()
        print('End')


absolute_path = "C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Aufgabe 3\\ArtKnowledgeGraph\\ArtKnowledgeGraph\\Data\\Art_Stories.csv"

A = ArtKnowledgeGraph(absolute_path, isPDF=False, topic= "Art Basel Fair")
A.process("I")