import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

if "vectors" not in st.session_state:
    # Initialize OllamaEmbeddings with the correct model
    st.session_state.embeddings = OllamaEmbeddings(model="llama2")  # Ensure "llama2" is pulled locally
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Elon_Musk")
    #st.session_state.docs = st.session_state.loader.load()

    # Split documents into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])

    # Create vector store
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retriever and retrieval chain
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit input for prompt
prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    start = time.time()
    response = retrieval_chain.invoke({"input": prompt_input})
    print("Response time:", time.time() - start)
    st.write(response['answer'])

    # Streamlit expander for document similarity search
    with st.expander("Document Similarity Search"):
        # Display relevant chunks
        if "context" in response:
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")