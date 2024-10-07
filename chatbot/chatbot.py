import re
import pinecone  # Import Pinecone library for vector database functionality
from pinecone import ServerlessSpec  # Specification for creating a Pinecone index in serverless mode
from langchain_community.document_loaders import PyPDFLoader  # To load PDF files as documents
from langchain_experimental.text_splitter import SemanticChunker  # Splits large texts into smaller chunks based on semantics
from langchain_community.chat_models import ChatOpenAI  # To interact with the GPT model from OpenAI
from langchain_pinecone import Pinecone  # A wrapper to use Pinecone with LangChain
from langchain.chains import RetrievalQA  # Retrieval-based question-answering chain
from pinecone.grpc import PineconeGRPC as Pinecone  # Pinecone gRPC client for high performance
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
import os
os.environ['PINECONE_API_KEY'] = 'cd4f53f7-4807-4868-81dd-c15760aa0b36'
os.environ['OPENAI_API_KEY'] = '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'
class ChatBot:
    def __init__(self, index_name):
        """
        Initialize the ChatBot class by setting API keys, setting up embeddings,
        and creating or connecting to a Pinecone index.

        Args:
            index_name (str): The name of the Pinecone index where document embeddings will be stored.
        """
        # OpenAI and Pinecone API keys are stored as private variables
        self.__openai_key = "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"
        self.__pinecone_key = "cd4f53f7-4807-4868-81dd-c15760aa0b36"

        # Ensure the API keys are provided, or raise an error
        if not self.__openai_key or not self.__pinecone_key:
            raise ValueError("Please set your OpenAI and Pinecone API keys.")

        # Set up OpenAI embeddings using the specified model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self.__openai_key)

        # Initialize a Pinecone instance using the Pinecone API key
        pc = Pinecone(api_key=self.__pinecone_key)

        # Clean up the index name by converting it to lowercase and removing non-alphanumeric characters
        self.index_name = re.sub(r'[^a-z0-9]', '', index_name.lower())
        self.dimension = 1536  # Dimension of embeddings to be stored
        self.metric = 'cosine'  # Metric for similarity search, here 'cosine' distance
        self.vectorstore = None  # Initialize vector store as None
        self.documents = []  # Empty list to store documents

        # Retrieve the list of existing indexes in Pinecone
        index_list = pc.list_indexes()

        # Extract index names from the retrieved index list
        existing_index_names = [index.name for index in index_list]

        # Check if the given index name already exists
        if self.index_name in existing_index_names:
            print(f"Index '{self.index_name}' already exists.")
        else:
            # Create the index if it doesn't already exist
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud='aws', region='us-east-1')  # AWS region specification for serverless index
            )
            print(f"Index '{self.index_name}' has been created.")

        # Connect to the Pinecone index for further operations
        self.index = pinecone.Index(self.index_name, 'aws')
        print(f"Connected to Pinecone index '{self.index_name}'.")

    def train(self, file_paths):
        """
        Train the chatbot by loading documents, splitting them into chunks,
        and indexing them in Pinecone.

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

        # Populate the Pinecone index with document chunks and embeddings
        self.vectorstore = PineconeVectorStore.from_documents(self.split_docs, self.embeddings, index_name=self.index_name)
        print(f"Populated Pinecone index: {self.index_name}")

# Ensure you have the correct import after updating the package

# Within your ask_question method
    def ask_question(self, query):
        """Use the vector store to retrieve answers to a query."""
        if not self.vectorstore:
            try:
                print("Vector store not initialized. Attempting to load the vector store.")
                self.load_vectorstore()
            except Exception as e:
                print(f"Failed to load vector store: {e}")
                return

        # Corrected import for ChatOpenAI from the updated package
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=self.__openai_key)

        # Assuming qa is a chain type
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.vectorstore.as_retriever())

        # Updated method call from __call__ to invoke
        answer = qa.invoke({"query": query})
        return answer

    


    def load_vectorstore(self):
        """Load the Pinecone vector store from an existing index."""
        try:
            self.vectorstore = PineconeVectorStore(index_name=self.index_name,embedding=self.embeddings)
            print(f"Loaded Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
