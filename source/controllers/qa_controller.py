from fastapi import APIRouter, HTTPException
from helper.file_processing import initialize_pinecone, initialize_embeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from helper.config import get_settings  

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
        
        # Set up retrieval and question-answering chain
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.as_retriever())

        # Get the answer to the query
        answer = qa_chain({"query": query})

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
