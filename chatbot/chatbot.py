import os
import re
from langchain_community.document_loaders import PyPDFLoader  # Updated import for PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker  # Splits large texts into smaller chunks based on semantics
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAI embeddings
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA  # Retrieval-based question-answering chain
from langchain_community.chat_models import ChatOpenAI  # To interact with the GPT model from OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChatBot:
    def __init__(self, index_name):
        """
        Initialize the ChatBot class by setting API keys, setting up embeddings,
        and creating or connecting to a FAISS index.

        Args:
            index_name (str): The name of the FAISS index file where document embeddings will be stored.
        """
        # Retrieve API keys from environment variables
        self.__openai_key = os.getenv('OPENAI_API_KEY')

        # Ensure the OpenAI API key is provided, or raise an error
        if not self.__openai_key:
            raise ValueError("Please set your OpenAI API key.")

        # Set up OpenAI embeddings using the specified model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=self.__openai_key)

        # Set index name and other attributes
        self.index_name = re.sub(r'[^a-z0-9]', '', index_name.lower()) + ".faiss"
        self.vectorstore = None
        self.documents = []

    def train(self, file_paths):
        """
        Train the chatbot by loading documents, splitting them into chunks,
        and indexing them in FAISS.

        Args:
            file_paths (str or list of str): A string path or list of file paths to PDF documents.
        """
        # If a single file path string is provided, convert it into a list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Validate that file_paths is a list of paths at this stage
        if not isinstance(file_paths, list):
            print("Error: file_paths must be either a string or a list of paths.")
            return

        # Filter and keep only PDF files from the provided paths
        pdf_paths = [path for path in file_paths if path.lower().endswith('.pdf')]

        # If no valid PDFs are found, display an error and exit
        if not pdf_paths:
            print("Error: No valid PDF files found.")
            return

        # Load each PDF file and append its contents to the documents list
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)  # Load PDF using PyPDFLoader
                print(f"Loading file: {path}")
                docs = loader.load()  # Load the documents from the PDF
                self.documents.extend(docs)  # Extend the documents list with the loaded content
                print(f"Loaded {len(docs)} documents from {path}.")
            except Exception as e:
                # Handle any exception during document loading
                print(f"Failed to load {path}: {e}")

        # If no documents were successfully loaded, exit the function
        if not self.documents:
            print("No documents were loaded. Exiting.")
            return

        # Split loaded documents into semantic chunks using the embeddings for better processing
        text_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        self.split_docs = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.split_docs)} chunks.")

        # Ensure that there are chunks to process
        if not self.split_docs:
            print("No documents to split. Please load and split documents first.")
            return

        # Convert split_docs to the format required by FAISS
        documents = [doc for doc in self.split_docs]

        # Populate the FAISS index with document chunks and embeddings
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print(f"Populated FAISS index with documents.")

    def ask_question(self, query):
        """Use the vector store to retrieve answers to a query."""
        if not self.vectorstore:
            print("Vector store not initialized. Please populate the index first.")
            return

        # Use the updated import for ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=self.__openai_key)

        # Assuming qa is a chain type
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.vectorstore.as_retriever())

        # Updated deprecated method `run` to `invoke`
        answer = qa({"query": query + " If a response cannot be formed strictly using the context, politely say you don’t have knowledge about that topic."})
        return answer
