#Creating a natural language-based chatbot in Python with LangChain and RAG to extract information from a collection of documents.
#From: Andrea D'Agostino, medium.com

import pandas as pd

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import streamlit as st

# Bibs f체r die Metriken (SSA) und Perplexit채t
from transformers import AutoTokenizer, AutoModelWithLMHead
from sklearn.preprocessing import normalize
from rouge import Rouge
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_dataset() -> pd.DataFrame:
    """
    Load dataset from file_path

    Returns:
        pd.DataFrame: Dataset
    """
    data_dir = "./data"
    dataset_name = "Art_Stories.csv"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df


def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> list:
    """
    Create chunks from the dataset

    Args:
        dataset (pd.DataFrame): Dataset
        chunk_size (int): Chunk size
        chunk_overlap (int): Chunk overlap

    Returns:
        list: List of chunks
    """
    page_content_column = "STORY_TEXT"
    text_chunks = DataFrameLoader(
        dataset, page_content_column=page_content_column
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    )
    # Add metadata to chunks to facilitate retrieval
    for doc in text_chunks:
        content = doc.page_content
        final_content = f"STORYTEXT: {content}"
        doc.page_content = final_content

    return text_chunks


def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Create or get vector store

    Args:
        chunks (list): List of chunks

    Returns:
        FAISS: Vector store
    """
    embeddings = HuggingFaceInstructEmbeddings()

    if not os.path.exists("./db/ArtStories"):
        print("CREATING DB")
        # Normalizing the vectors before storing could optimize the chatBot response at least a tiny bit.
        vectors = [normalize(embeddings.embed(chunk)) for chunk in chunks]
        vectorstore = FAISS.from_arrays(vectors)
        vectorstore.save_local("./db/ArtStories")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db/ArtStories", embeddings)

    return vectorstore


def get_conversation_chain(vector_store: FAISS, system_message: str, human_message: str) -> ConversationalRetrievalChain:
    """
    Get the chatbot conversation chain

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.5, "max_length": 64})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain


# Aufgabe 2 Perplexit채t)
def calculate_perplexity(response):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    input_ids = tokenizer.encode(response, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        # calculate the perplexity out of the entropy-loss
        perplexity = torch.exp(loss)
    return perplexity.item()


# Aufgabe 2 (SSA) )
# Initialize Rouge so we are able to calculate the sesibleness and specificity
rouge = Rouge()


def calculate_sensibleness(user_question, response):
    rouge_scores = rouge.get_scores(user_question, response)
    rouge_1_recall = rouge_scores[0]['rouge-1']['r']
    rouge_2_recall = rouge_scores[0]['rouge-2']['r']
    return (rouge_1_recall + rouge_2_recall) / 2


def calculate_specificity(user_question, response):
    rouge_scores = rouge.get_scores(response, user_question)
    rouge_1_recall = rouge_scores[0]['rouge-1']['r']
    rouge_2_recall = rouge_scores[0]['rouge-2']['r']
    return (rouge_1_recall + rouge_2_recall) / 2


def calculate_ssa(user_question, response):
    sensibleness = calculate_sensibleness(user_question, response)
    specificity = calculate_specificity(user_question, response)
    return (sensibleness + specificity) / 2


# We decided to extract the function in the method for better understanding, so we could find something to optimize
def handle_style_and_responses(user_question: str) -> None:
    # Generate response
    response = generate_response(user_question)

    # Display conversation history
    for i, message in enumerate(response["chat_history"]):
        if i % 2 == 0:
            display_user_message(message)
        else:
            display_chatbot_message(message)


# Extracted out of handle_style_and_responses
def generate_response(user_question: str) -> dict:
    response = st.session_state.conversation({"question": user_question})
    return response


# Extracted out of handle_style_and_responses
def display_user_message(message) -> None:
    # Display user message
    human_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px;"
    st.markdown(
        f"<p style='text-align: right;'><b>User</b></p> <p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
        unsafe_allow_html=True,
    )


# Extracted out of handle_style_and_responses
def display_chatbot_message(user_question, message) -> None:
    # Display chatbot message
    chatbot_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px;"
    ssa_score = calculate_ssa(user_question, message.content)
    perplexity_score = calculate_perplexity(message.content)
    st.markdown(
        f"<p style='text-align: left;'>SSA: {ssa_score}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align: left;'>Perplexity: {perplexity_score}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
        unsafe_allow_html=True,
    )


# Extracted out of handle_style_and_responses
def display_conversation_history(user_question, response: dict) -> None:
    for i, message in enumerate(response["chat_history"]):
        if i % 2 == 0:
            display_user_message(message)
        else:
            display_chatbot_message(user_question, message)


def main():
    load_dotenv()
    df = load_dataset()
    chunks = create_chunks(df, chunk_size=1000, chunk_overlap=10)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.

        You should never answer a question with a question, and you should always respond with the most relevant documentation page.

        Do not answer questions that are not about the LangChain library or project.

        Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
        {context}
        """
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    # Initialize session state variables
    if "key" not in st.session_state:
        st.session_state.key = 'value'
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(
        page_title="Art Chatbot",
        page_icon=":books:",
    )

    st.title("Art Chatbot")
    st.subheader("Chat with art stories!")
    st.markdown(
        """
        This chatbot was created to answer questions about art stories.
        Ask a question and the chatbot will respond with the most relevant art story.
        """
    )
    st.image("https://images.unsplash.com/photo-1485827404703-89b55fcc595e")

    user_question = st.text_input("Ask your question")
    with st.spinner("Processing..."):
        if user_question:
            response = generate_response(user_question)
            display_conversation_history(user_question, response)

    # Load or create conversation chain
    if not st.session_state.conversation:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store, system_message_prompt, human_message_prompt
        )


if __name__ == "__main__":
    main()


    # Aufgabe 4.1, wir gehen davon aus das wir nach einer Anfrage jeweils nach <role> dementsprechende Ergebnisse
    # erwarten.
    # (Die Aufgabe wird als Kunde betrachtet, ohne Wissen 체ber die Datenbank)
    """
    | Prompt | As a <role> | I want <goal> | so that <benefit> | Acceptance Criteria |
    |--------|--------------|---------------|-------------------|---------------------|
    | Could you provide me information about Art from Africa | As an art collector (private collector) | I want to receive information about the material and technique of an artwork and of course the culturel aspect of it | So that I can somehow connect with the artwork and decide to buy recommended ones or not | The chatbot provides accurate information about the material and technique of the artwork and the cultural background. |
    | Can you provide me some other Artworks related to France, more likely Paris | As an art collector (museum in france) | I want to be able to ask the chatbot for recommendations on similar artists or artworks so that I can find new art pieces to put into the museum| To expand my knowledge of various art styles and artists | The chatbot provides relevant recommendations based on the art collector's interests. |
    | Hey I make art about mostly landscapes like Bob Ross, can you inspire me with some for new ideas | As an art maker (artist) | I want to get inspired by new art or art stories | So that I can draw something new or even out of my normal habitat | The chatbot should in this case provide logical recommandation about landscapes... |
    
    """
    # Aufgabe 4.2, Die Metriken wurden berechnet und es kamen plausible werte f체r diese raus. F체r die Optimierung wurde
    # der Code sauber aufgeteilt um optimierungsstellen erkennen zu k철nnen. Es wurde die vektoren erstellung
    # "optimiert",
    # uns war wird dieser Normalisiert.

    # Aufgabe 4.3, beurteilen von weiteren Optimierungspotenzialien
    """
    Anstelle von Rogue oder BLEU oder anderen Metriken sollte eine selbsterstellte Metrik zur Berechnung von SSA genutzt
    werden, weil die selbsterstellte Metrik auf unseren chat-Bot optimiert werden k철nnen. Es gibt vier Antworttypen extraktiv,
    abstrakt, ja-nein und multiple-choiche. Unsere selbsterstellte Metrik k철nnte beispielsweise f체r extraktive Antworten optimiert
    sein, wenn unser chat-Bot nur extraktive Antworten gibt.
    Unser chat-Bot soll kontextbezogene Antworten (passend zu einer z.B. Art story) liefern. Das heist der chat-Bot soll daf체r
    optimiert sein und kein "allesk철nner" sein --> Bezogen auf eine Kunstmesse.
    Siehe: https://fbi.h-da.de/fileadmin/Personen/fbi1119/Michel-Masterarbeit.pdf
    """