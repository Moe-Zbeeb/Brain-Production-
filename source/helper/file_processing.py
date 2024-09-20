# helper/file_processing.py

import os
import re
from langchain.document_loaders import PyPDFLoader  # For PDF file loading
from langchain_experimental.text_splitter import SemanticChunker  # For chunking text
from langchain.embeddings.openai import OpenAIEmbeddings  # For OpenAI embeddings
from langchain_pinecone import PineconeVectorStore
from helper.config import get_settings
import pinecone

# Initialize OpenAI Embeddings and Pinecone
def initialize_embeddings():
    settings = get_settings()
    openai_key = settings.OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_key)
    return embeddings

# Initialize Pinecone index
def initialize_pinecone():
    settings = get_settings()
    pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
    return pinecone

# Function to process files and create embeddings
def process_file(file_path, file_type):
    settings = get_settings()
    
    # Initialize OpenAI embeddings
    embeddings = initialize_embeddings()

    # Load documents based on file type
    documents = []
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    
    # Implement loading for PowerPoint and TXT later...
    
    # Split documents into smaller chunks for embedding
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    split_docs = text_splitter.split_documents(documents)

    # Ensure Pinecone index is created or connected
    index_name = settings.PINECONE_INDEX_NAME
    pinecone_instance = initialize_pinecone()
    if index_name not in pinecone_instance.list_indexes():
        pinecone_instance.create_index(name=index_name, dimension=settings.PINECONE_DIMENSION, metric=settings.PINECONE_METRIC)

    # Connect to the index and upload embeddings
    vectorstore = PineconeVectorStore.from_documents(split_docs, embeddings, index_name=index_name)
    
    return vectorstore
