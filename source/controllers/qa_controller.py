# controllers/qa_controller.py
from fastapi import APIRouter, HTTPException
from helper.file_processing import initialize_pinecone, initialize_embeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

qa_router = APIRouter(
    prefix="/api/v1/qa",
    tags=["Question Answering"]
)

@qa_router.get("/ask")
async def ask_question(query: str, index_name: str):
    try:
        # Validate the index name provided by the user
        if not index_name or not index_name.isalnum():
            raise HTTPException(status_code=400, detail="Invalid index name. It must be alphanumeric.")

        # Initialize Pinecone and embeddings
        pinecone_instance = initialize_pinecone()
        embeddings = initialize_embeddings()

        # Check if the index exists
        if index_name not in pinecone_instance.list_indexes():
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found.")

        # Connect to the specified Pinecone index
        index = pinecone_instance.Index(index_name)

        # Set up retrieval and question-answering chain
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=embeddings.openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.as_retriever())

        # Get the answer to the query
        answer = qa_chain({"query": query})

        return {"answer": answer["text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
