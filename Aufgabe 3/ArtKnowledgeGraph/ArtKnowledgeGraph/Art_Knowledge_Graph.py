import warnings
warnings.filterwarnings("ignore")

## for data
import pandas as pd  #1.1.5
import numpy as np  #1.21.0

## for plotting
import matplotlib.pyplot as plt  #3.3.2
import seaborn as sns  #0.11.1

## for text
import wikipediaapi  #0.5.8
#import wikipedia  #0.5.8
import nltk  #3.8.1
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import re   

## for nlp
import spacy  #3.5.0
from spacy import displacy
import textacy  #0.12.0

## for graph
import networkx as nx  #3.0 (also pygraphviz==1.10)
#import pygraphviz
import plotly.graph_objs as go  #5.1.0

## for timeline
import dateparser #1.1.7

#Read PDF-files
import PyPDF2
#import pdfquery
#from pdfquery import PDFQuery

#Data source

#Target data: Art Stories
#Example: Read sample pdf-file via PyPDF2
# creating a pdf file object
pdfFileObj = open('Data\Example\The_Art_Basel_and_UBS_Survey_of_Global_Collecting_in_2023_EN.pdf', 'rb') 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)
number_of_pages = len(pdfReader.pages)
#print(len(pdfReader.pages))

page_content = ''
#Loop through pdf-pages
for page_number in range(number_of_pages):
    # create page object
    pageObj = pdfReader.pages[page_number]
    # extracting text from page
    page_content += pageObj.extract_text()

#    page = read_pdf.getPage(page_number)
#    page_content += page.extractText()
#print(pageObj.extract_text())

# closing the pdf file object
pdfFileObj.close()

txt = page_content

topic = "Art Basel Fair"
#topic = "Russo-Ukrainian War"

#wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
##wiki = wikipedia.set_lang('en')
#page = wiki.page(topic)
#txt = page.text[:page.text.find("See also")]
#txt[0:500] + " ..."


'''
Compute n-grams frequency with nltk tokenizer.
:parameter
    :param txt: text
    :param ngrams: int or list - 1 for unigrams, 2 for bigrams, [1,2] for both
    :param top: num - plot the top frequent words
:return
    dtf_count: dtf with word frequency
'''
def word_freq(txt, ngrams=[1,2,3], top=10, figsize=(10,7)):
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
#    plt.show()

    return dtf_freq


# Find most common words in text
_ = word_freq(txt, ngrams=[3], top=30, figsize=(10,7))

#Model
#python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
doc = nlp(txt)

#Sentence Segmentation
lst_docs = [sent for sent in doc.sents]
print("tot sentences:", len(lst_docs))

i = 3
lst_docs[i]

print(lst_docs[i], "\n---")

for token in lst_docs[i]:
    print(token.text, "-->", "pos: "+token.pos_, "|", "dep: "+token.dep_, "")


displacy.render(lst_docs[i], style="dep", options={"distance":100})


for tag in lst_docs[i].ents:
    print(tag.text, f"({tag.label_})")

displacy.render(lst_docs[i], style="ent")


#2 - Entities, Relations, Attributes
#Entities Extraction

## Using POS/DEP
def extract_entities(doc):
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


lst_entities = [extract_entities(i) for i in lst_docs]
lst_entities[i]


#Relation Extraction
## Using Matcher
def extract_relation(doc, nlp):
    matcher = spacy.matcher.Matcher(nlp.vocab)
    p1 = [{'DEP':'ROOT'}, 
          {'DEP':'prep', 'OP':"?"},
          {'DEP':'agent', 'OP':"?"},
          {'POS':'ADJ', 'OP':"?"}] 
    matcher.add(key="matching_1", patterns=[p1]) 
    matches = matcher(doc)
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]] 
    return span.text


lst_relations = [extract_relation(i,nlp) for i in lst_docs]
lst_relations[i]

#Attribute Extraction
## Using NER
lst_attr = []
for x in lst_docs:
    attr = ""
    for tag in x.ents:
        attr = attr+tag.text if tag.label_=="DATE" else attr+""
    lst_attr.append(attr)

lst_attr[i]

#Summary
dtf = pd.DataFrame({"text":[doc.text for doc in lst_docs],
                    "entity":[i[0] for i in lst_entities],
                    "relation":lst_relations,
                    "object":[i[1] for i in lst_entities],
                    "attribute":lst_attr
                   })
print(dtf.head(3))


#3 - Textacy
#Entities and Relations
dic = {"id":[], "text":[], "entity":[], "relation":[], "object":[]}

for n,sentence in enumerate(lst_docs):
    lst_generators = list(textacy.extract.subject_verb_object_triples(sentence))  
    for sent in lst_generators:
        subj = "_".join(map(str, sent.subject))
        obj  = "_".join(map(str, sent.object))
        relation = "_".join(map(str, sent.verb))
        dic["id"].append(n)
        dic["text"].append(sentence.text)
        dic["entity"].append(subj)
        dic["object"].append(obj)
        dic["relation"].append(relation)

dtf = pd.DataFrame(dic)
dtf[dtf["id"]==i]

dtf = dtf[dtf["object"].str.len() < 20]

#Attributes
## Date
attribute = "date"
dic = {"id":[], "text":[], attribute:[]}

for n,sentence in enumerate(lst_docs):
    lst = list(textacy.extract.entities(sentence, include_types={"DATE"}))
    if len(lst) > 0:
        for attr in lst:
            dic["id"].append(n)
            dic["text"].append(sentence.text)
            dic[attribute].append(str(attr))
    else:
        dic["id"].append(n)
        dic["text"].append(sentence.text)
        dic[attribute].append(np.nan)

dtf_att = pd.DataFrame(dic)
dtf_att = dtf_att[~dtf_att[attribute].isna()]
dtf_att[dtf_att["id"]==i]


## Location
attribute = "location"
dic = {"id":[], "text":[], attribute:[]}

for n,sentence in enumerate(lst_docs):
    lst = list(textacy.extract.entities(sentence, include_types={"LOC","GPE"}))
    if len(lst) > 0:
        for attr in lst:
            dic["id"].append(n)
            dic["text"].append(sentence.text)
            dic[attribute].append(str(attr))
    else:
        dic["id"].append(n)
        dic["text"].append(sentence.text)
        dic[attribute].append(np.nan)

dtf_att_2 = pd.DataFrame(dic)
dtf_att_2 = dtf_att_2[~dtf_att_2[attribute].isna()]

dtf_att_2[dtf_att_2["id"]==i]

#4 - Network Graph
#Networkx
## full
G = nx.from_pandas_edgelist(dtf, source="entity", target="object", edge_attr="relation", create_using=nx.DiGraph())

plt.figure(figsize=(15,10))

#pos = nx.nx_agraph.graphviz_layout(G, prog="fdp")
pos = nx.spring_layout(G, k=1)

node_color = "skyblue"
edge_color = "black"

nx.draw(G, pos=pos, with_labels=True, node_color=node_color, edge_color=edge_color, cmap=plt.cm.Dark2, 
        node_size=2000, connectionstyle='arc3,rad=0.1')

nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(G,'relation'),
                             font_size=12, font_color='black', alpha=0.6)
#Plot graph of whole dataset:
#plt.show()


## top
print("Entity count:")
print(dtf["entity"].value_counts().head())

## filter
f = "UBS"
tmp = dtf[(dtf["entity"]==f) | (dtf["object"]==f)]

G = nx.from_pandas_edgelist(tmp, source="entity", target="object", edge_attr="relation", create_using=nx.DiGraph())


plt.figure(figsize=(15,10))

pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
#pos = nx.spring_layout(G, k=2.5)

node_color = ["red" if node==f else "skyblue" for node in G.nodes]
edge_color = ["red" if edge[0]==f else "black" for edge in G.edges]

nx.draw(G, pos=pos, with_labels=True, node_color=node_color, edge_color=edge_color, cmap=plt.cm.Dark2, 
        node_size=2000, node_shape="o", connectionstyle='arc3,rad=0.1')

nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(G,'relation'),
                             font_size=12, font_color='black', alpha=0.6)

#Plot graph of filtered dataset
#plt.show()


## 3D
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection="3d")
pos = nx.spring_layout(G, k=2.5, dim=3)

nodes = np.array([pos[v] for v in sorted(G) if v!=f])
center_node = np.array([pos[v] for v in sorted(G) if v==f])

edges = np.array([(pos[u],pos[v]) for u,v in G.edges() if v!=f])
center_edges = np.array([(pos[u],pos[v]) for u,v in G.edges() if v==f])

ax.scatter(*nodes.T, s=200, ec="w", c="skyblue", alpha=0.5)
ax.scatter(*center_node.T, s=200, c="red", alpha=0.5)

for link in edges:
    ax.plot(*link.T, color="grey", lw=0.5)
for link in center_edges:
    ax.plot(*link.T, color="red", lw=0.5)
    
for v in sorted(G):
    ax.text(*pos[v].T, s=v)
for u,v in G.edges():
    attr = nx.get_edge_attributes(G, "relation")[(u,v)]
    ax.text(*((pos[u]+pos[v])/2).T, s=attr)

ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], yticklabels=[], zticklabels=[])
ax.grid(False)
for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    dim.set_ticks([])
#plt.show()


#Plotly
## setup
pos = nx.spring_layout(G, k=0.2)
for n,p in pos.items():
    G.nodes[n]['pos'] = p
    
## links
edge_x, edge_y = [], []
arrows = []
for n,edge in enumerate(G.edges()):
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    arrows.append([[x0,y0],[x1,y1]])

edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', showlegend=False)

## nodes
node_x, node_y = [], []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', showlegend=False,
                        marker={"showscale":False, "colorscale":'YlGnBu', "reversescale":True, 
                                "size":10, "line_width":2})

## add details
link_text, node_text, node_color, node_size = [], [], [], []
for adjacencies in G.adjacency():
    node_text.append(adjacencies[0])
    for dic in adjacencies[1].values():
        link_text.append(dic["relation"])

node_trace.text = node_text
edge_trace.text = link_text

## layout
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(title=topic, showlegend=True, plot_bgcolor='white', 
                  hovermode='closest', width=800, height=800,
                  xaxis={"visible":False}, yaxis={"visible":False})

#fig.show()

#5 - Timeline
#Parse data
def utils_parsetime(txt):
    x = re.match(r'.*([1-3][0-9]{3})', txt) #<-- check if there is a year
    if x is not None:
        try:
            dt = dateparser.parse(txt)
        except:
            dt = np.nan
    else:
        dt = np.nan
    return dt


dtf_att["dt"] = dtf_att["date"].apply(lambda x: utils_parsetime(x))
dtf_att[dtf_att["id"]==i]


## Merge
tmp = dtf.copy()
tmp["y"] = tmp["entity"]+" "+tmp["relation"]+" "+tmp["object"]

dtf_att = dtf_att.merge(tmp[["id","y"]], how="left", on="id")
dtf_att = dtf_att[~dtf_att["y"].isna()].sort_values("dt", ascending=True).drop_duplicates("y", keep='first')
dtf_att.head()


## Full
dates = dtf_att["dt"].values
names = dtf_att["y"].values
l = [10,-10, 8,-8, 6,-6, 4,-4, 2,-2]
levels = np.tile(l, int(np.ceil(len(dates)/len(l))))[:len(dates)]

fig, ax = plt.subplots(figsize=(20,10))
ax.set(title=topic, yticks=[], yticklabels=[])

ax.vlines(dates, ymin=0, ymax=levels, color="tab:red")
ax.plot(dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w")

for d,l,r in zip(dates,levels,names):
    ax.annotate(r, xy=(d,l), xytext=(-3, np.sign(l)*3), textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="bottom" if l>0 else "top")
plt.xticks(rotation=90) 
#Plot full timeline
#plt.show()


## Filter
yyyy = "2023"
dates = dtf_att[dtf_att["dt"]>yyyy]["dt"].values
names = dtf_att[dtf_att["dt"]>yyyy]["y"].values
l = [10,-10, 8,-8, 6,-6, 4,-4, 2,-2]
levels = np.tile(l, int(np.ceil(len(dates)/len(l))))[:len(dates)]

fig, ax = plt.subplots(figsize=(20,10))
ax.set(title=topic, yticks=[], yticklabels=[])

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