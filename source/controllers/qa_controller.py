# controllers/qa_controller.py

from fastapi import APIRouter, HTTPException
from helper.file_processing import initialize_pinecone, initialize_embeddings
from helper.config import get_settings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

qa_router = APIRouter(
    prefix="/api/v1/qa",
    tags=["Question Answering"]
)

@qa_router.get("/ask")
async def ask_question(query: str):
    try:
        # Initialize Pinecone and embeddings
        pinecone_instance = initialize_pinecone()
        embeddings = initialize_embeddings()

        # Connect to Pinecone index
        settings = get_settings()
        index_name = settings.PINECONE_INDEX_NAME
        index = pinecone_instance.Index(index_name)
        
        # Set up retrieval and question-answering chain with strict prompt constraints
        llm = ChatOpenAI(
            model="gpt-4", 
            temperature=0,  # Lower temperature to reduce creative output
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Define a stricter prompt to keep responses on-topic
        strict_prompt = (
            "You are a highly knowledgeable assistant that can only answer based on the provided context. "
            "Do not provide information that is not directly supported by the context below. "
            "If you don't know the answer, respond with 'I don't have the knowledge about that topic.'"
        )

        # Set up the QA chain with the strict prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=index.as_retriever(),
            verbose=True,
            initial_prompt=strict_prompt
        )

        # Execute the QA chain with the query
        response = qa_chain({"query": query})

        # Check if the response is relevant or indicates a lack of information
        if "I don't have the knowledge about that topic." in response["text"]:
            return {"answer": "The answer is not available in the current database context."}
        
        return {"answer": response["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
